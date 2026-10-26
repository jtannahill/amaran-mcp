// amaran-systemtap: stream what this Mac is playing to stdout, for music mode.
//
// Output: one line "RATE <hz>\n", then raw little-endian float32 mono samples
// until SIGTERM or SIGINT. Exit code 3, with a message on stderr, if the
// system audio tap cannot be set up. Needs macOS 14.2 or later.
//
// The tap listens to every process's output (AirPlay Receiver included) after
// mixing, so it never opens a microphone and never touches AirPods' input.

import AudioToolbox
import CoreAudio
import Foundation

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

// 3. Write each buffer to stdout as mono float32.
let header = "RATE \(Int(format.mSampleRate))\n"
FileHandle.standardOutput.write(header.data(using: .utf8)!)

var mono = [Float32](repeating: 0, count: 8192)
var procID: AudioDeviceIOProcID?
check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { _, input, _, _, _ in
    let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
    guard let first = buffers.first, let firstData = first.mData else { return }
    let channels = Int(first.mNumberChannels)
    if buffers.count == 1 && channels == 1 {
        fwrite(firstData, 1, Int(first.mDataByteSize), stdout)
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
