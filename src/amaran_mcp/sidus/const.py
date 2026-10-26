"""The constants protocol.py needs from amaran-hacs's const module."""

SIDUS_ACCESS_OPCODE = 0x26

# An empty reject list makes the proxy forward every mesh message to us, which
# is how the light's status and battery reports arrive.
PROXY_FILTER_TYPE_REJECT = 0x01
