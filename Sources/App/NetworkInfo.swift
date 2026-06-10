import Foundation

#if canImport(Darwin)
import Darwin
#endif

/// Small helper to discover the device's LAN IPv4 address so the UI can show
/// other machines on the network where to point their OpenAI client.
enum NetworkInfo {

    /// Returns the first non-loopback IPv4 address, preferring the Wi-Fi
    /// interface (`en0`). Returns `nil` if none is found (e.g. no Wi-Fi).
    static func wifiIPv4Address() -> String? {
        var preferred: String?
        var fallback: String?

        var ifaddrPtr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddrPtr) == 0, let first = ifaddrPtr else {
            return nil
        }
        defer { freeifaddrs(ifaddrPtr) }

        var ptr: UnsafeMutablePointer<ifaddrs>? = first
        while let current = ptr {
            let interface = current.pointee
            defer { ptr = interface.ifa_next }

            // getifaddrs can return entries with a NULL ifa_addr — guard before
            // dereferencing, otherwise this nil-derefs and crashes.
            guard let addr = interface.ifa_addr else { continue }
            let family = addr.pointee.sa_family

            if family == UInt8(AF_INET) {
                let name = String(cString: interface.ifa_name)
                var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                let result = getnameinfo(
                    addr,
                    socklen_t(addr.pointee.sa_len),
                    &host, socklen_t(host.count),
                    nil, 0,
                    NI_NUMERICHOST
                )
                if result == 0 {
                    let ip = String(cString: host)
                    if ip != "127.0.0.1" {
                        if name == "en0" {
                            preferred = ip
                        } else if fallback == nil {
                            fallback = ip
                        }
                    }
                }
            }
        }

        return preferred ?? fallback
    }
}
