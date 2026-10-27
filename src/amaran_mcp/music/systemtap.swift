// amaran-systemtap: stream what this Mac is playing to stdout, for music mode.
//
// Output: one line "RATE <hz>\n", then raw little-endian float32 mono samples
// until SIGTERM or SIGINT. Exit code 3, with a message on stderr, if the
// system audio tap cannot be set up. Needs macOS 14.2 or later.
//
// With --classify-fd N it also runs Apple's on-device sound classifier
// (SoundAnalysis) on the same audio and writes about twice a second, on file
// descriptor N, one JSON line of confidences: singing, speech, and so on.
//
// The tap listens to every process's output (AirPlay Receiver included) after
// mixing, so it never opens a microphone and never touches AirPods' input.

import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation
import SoundAnalysis

func fail(_ message: String, _ status: OSStatus) -> Never {
    FileHandle.standardError.write("amaran-systemtap: \(message) (OSStatus \(status))\n".data(using: .utf8)!)
    exit(3)
}

func check(_ status: OSStatus, _ message: String) {
    if status != noErr { fail(message, status) }
}

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
}

func defaultOutputUID() -> String {
    var where_ = address(kAudioHardwarePropertyDefaultSystemOutputDevice)
    var device = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &where_, 0, nil, &size, &device),
          "could not find the output device")
    var uidAddress = address(kAudioDevicePropertyDeviceUID)
    var uid: Unmanaged<CFString>?
    size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    check(AudioObjectGetPropertyData(device, &uidAddress, 0, nil, &size, &uid),
          "could not read the output device")
    return uid!.takeRetainedValue() as String
}

var classifyFD: Int32? = nil
var classifyFile: String? = nil
var arguments = CommandLine.arguments.dropFirst()
while let argument = arguments.popFirst() {
    if argument == "--classify-fd", let value = arguments.popFirst(), let fd = Int32(value) {
        classifyFD = fd
    } else if argument == "--classify-file", let value = arguments.popFirst() {
        classifyFile = value
    } else {
        FileHandle.standardError.write("amaran-systemtap: unknown option \(argument)\n".data(using: .utf8)!)
        exit(2)
    }
}

/// Writes classifier results as JSON lines on the side channel.
final class ClassWriter: NSObject, SNResultsObserving {
    static let labels = ["singing", "choir_singing", "speech", "rapping", "cheering", "crowd", "applause"]
    let handle: FileHandle

    init(fd: Int32) { handle = FileHandle(fileDescriptor: fd, closeOnDealloc: false) }

    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let result = result as? SNClassificationResult else { return }
        var parts = [String(format: "\"t\": %.2f", result.timeRange.start.seconds)]
        for label in ClassWriter.labels {
            let confidence = result.classification(forIdentifier: label)?.confidence ?? 0
            parts.append(String(format: "\"%@\": %.3f", label, confidence))
        }
        try? handle.write(contentsOf: Data(("{" + parts.joined(separator: ", ") + "}\n").utf8))
    }

    func request(_ request: SNRequest, didFailWithError error: Error) {
        FileHandle.standardError.write("amaran-systemtap: classifier stopped: \(error)\n".data(using: .utf8)!)
    }
}

// File mode (for training): classify an audio file, print the same JSON lines
// on stdout, and exit. No tap, no capture permission needed.
if let path = classifyFile {
    do {
        let analyzer = try SNAudioFileAnalyzer(url: URL(fileURLWithPath: path))
        let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
        request.windowDuration = CMTime(seconds: 1.5, preferredTimescale: 48000)
        request.overlapFactor = 0.66
        let writer = ClassWriter(fd: STDOUT_FILENO)
        try analyzer.add(request, withObserver: writer)
        analyzer.analyze()
        exit(0)
    } catch {
        FileHandle.standardError.write("amaran-systemtap: cannot classify \(path): \(error)\n".data(using: .utf8)!)
        exit(3)
    }
}

// 1. A private tap on all processes' output, mixed to mono.
let tap = CATapDescription(monoGlobalTapButExcludeProcesses: [])
tap.name = "amaran-mcp music mode"
tap.isPrivate = true
tap.muteBehavior = .unmuted
var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(tap, &tapID), "could not create the system audio tap")

var format = AudioStreamBasicDescription()
var formatAddress = address(kAudioTapPropertyFormat)
var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
check(AudioObjectGetPropertyData(tapID, &formatAddress, 0, nil, &formatSize, &format),
      "could not read the tap format")
if format.mFormatFlags & kAudioFormatFlagIsFloat == 0 || format.mBitsPerChannel != 32 {
    fail("the tap delivers a non-float32 format", -1)
}

// 2. A private aggregate device that carries the tap, clocked by the output.
let outputUID = defaultOutputUID()
let aggregate: [String: Any] = [
    kAudioAggregateDeviceNameKey: "amaran-mcp music mode",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapDriftCompensationKey: true,
        kAudioSubTapUIDKey: tap.uuid.uuidString,
    ]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID),
      "could not create the capture device")

// 3. Write each buffer to stdout as mono float32, and (optionally) feed a copy
// to the classifier on its own queue so the audio callback never waits.
let header = "RATE \(Int(format.mSampleRate))\n"
FileHandle.standardOutput.write(header.data(using: .utf8)!)

let analysisQueue = DispatchQueue(label: "amaran-systemtap.classify")
var streamAnalyzer: SNAudioStreamAnalyzer? = nil
var classWriter: ClassWriter? = nil
var framePosition: AVAudioFramePosition = 0
let monoFormat = AVAudioFormat(standardFormatWithSampleRate: format.mSampleRate, channels: 1)!
if let fd = classifyFD {
    do {
        let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
        request.windowDuration = CMTime(seconds: 1.5, preferredTimescale: 48000)
        request.overlapFactor = 0.66
        let analyzer = SNAudioStreamAnalyzer(format: monoFormat)
        let writer = ClassWriter(fd: fd)
        try analyzer.add(request, withObserver: writer)
        (streamAnalyzer, classWriter) = (analyzer, writer)
    } catch {
        FileHandle.standardError.write("amaran-systemtap: no classifier (\(error)); audio only\n".data(using: .utf8)!)
    }
}

func classify(_ samples: UnsafePointer<Float32>, _ frames: Int) {
    guard let analyzer = streamAnalyzer, frames > 0 else { return }
    let copy = Data(bytes: samples, count: frames * 4)
    analysisQueue.async {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: AVAudioFrameCount(frames)) else { return }
        buffer.frameLength = AVAudioFrameCount(frames)
        copy.withUnsafeBytes { raw in
            buffer.floatChannelData![0].update(from: raw.bindMemory(to: Float32.self).baseAddress!, count: frames)
        }
        analyzer.analyze(buffer, atAudioFramePosition: framePosition)
        framePosition += AVAudioFramePosition(frames)
    }
}

var mono = [Float32](repeating: 0, count: 8192)
var procID: AudioDeviceIOProcID?
check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { _, input, _, _, _ in
    let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
    guard let first = buffers.first, let firstData = first.mData else { return }
    let channels = Int(first.mNumberChannels)
    if buffers.count == 1 && channels == 1 {
        fwrite(firstData, 1, Int(first.mDataByteSize), stdout)
        classify(firstData.assumingMemoryBound(to: Float32.self), Int(first.mDataByteSize) / 4)
    } else {
        // Interleaved (one buffer, many channels) or one buffer per channel.
        let interleaved = buffers.count == 1
        let frames = Int(first.mDataByteSize) / (4 * max(channels, 1))
        let width = interleaved ? channels : buffers.count
        if mono.count < frames { mono = [Float32](repeating: 0, count: frames) }
        for frame in 0..<frames {
            var sum: Float32 = 0
            for channel in 0..<width {
                if interleaved {
                    sum += firstData.assumingMemoryBound(to: Float32.self)[frame * channels + channel]
                } else if let data = buffers[channel].mData {
                    sum += data.assumingMemoryBound(to: Float32.self)[frame]
                }
            }
            mono[frame] = sum / Float32(width)
        }
        mono.withUnsafeBytes { raw in _ = fwrite(raw.baseAddress, 4, frames, stdout) }
        mono.withUnsafeBufferPointer { classify($0.baseAddress!, frames) }
    }
    fflush(stdout)
}, "could not start capture")
check(AudioDeviceStart(aggregateID, procID), "could not start capture")

// 4. Clean up on SIGTERM / SIGINT. A broken pipe (reader gone) ends the
// process by default, and the private tap and device go with it.
func stop() -> Never {
    AudioDeviceStop(aggregateID, procID)
    if let procID { AudioDeviceDestroyIOProcID(aggregateID, procID) }
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    exit(0)
}

var signalSources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { stop() }
    source.resume()
    signalSources.append(source)
}
dispatchMain()
