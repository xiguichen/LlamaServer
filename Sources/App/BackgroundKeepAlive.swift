import Foundation
import AVFoundation

/// Keeps the app (and therefore the HTTP server) running while it is in the
/// background.
///
/// iOS suspends an app shortly after it is backgrounded, which freezes the
/// Telegraph run loop so incoming requests hang until the app is foregrounded
/// again. Declaring the `audio` background mode in Info.plist is necessary but
/// **not** sufficient — the system only keeps the process alive while an
/// `AVAudioSession` is actively playing audio. We therefore loop a short,
/// completely silent buffer through an active session.
///
/// The session uses `.mixWithOthers`, so it never interrupts or ducks the
/// user's music/podcast and produces no audible output.
final class BackgroundKeepAlive {

    private var player: AVAudioPlayer?

    /// Begin holding the app awake. Idempotent.
    func start() {
        guard player == nil else { return }
        do {
            let session = AVAudioSession.sharedInstance()
            // `.playback` is the only category honoured for background audio.
            // `.mixWithOthers` keeps us inaudible alongside other audio.
            try session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
            try session.setActive(true)

            let data = Self.makeSilentWavData()
            let player = try AVAudioPlayer(data: data)
            player.numberOfLoops = -1   // loop forever
            player.volume = 0           // belt-and-braces: truly silent
            player.prepareToPlay()
            player.play()
            self.player = player
            FileLogger.shared.info("background keep-alive started (silent audio session active)")
        } catch {
            FileLogger.shared.warn("background keep-alive failed: \(error.localizedDescription)")
        }
    }

    /// Stop holding the app awake and release the audio session.
    func stop() {
        guard player != nil else { return }
        player?.stop()
        player = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        FileLogger.shared.info("background keep-alive stopped")
    }

    // MARK: - Silent audio source

    /// Builds a minimal in-memory mono 16-bit PCM WAV containing a fraction of
    /// a second of silence. Generated in code so we don't have to ship an audio
    /// asset in the bundle.
    private static func makeSilentWavData() -> Data {
        let sampleRate = 8000
        let numChannels = 1
        let bitsPerSample = 16
        let durationFrames = sampleRate / 10            // 0.1 s
        let blockAlign = numChannels * bitsPerSample / 8
        let dataBytes = durationFrames * blockAlign
        let byteRate = sampleRate * blockAlign

        var data = Data()
        func appendString(_ s: String) { data.append(contentsOf: s.utf8) }
        func appendUInt32LE(_ v: UInt32) {
            data.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
                                     UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff)])
        }
        func appendUInt16LE(_ v: UInt16) {
            data.append(contentsOf: [UInt8(v & 0xff), UInt8((v >> 8) & 0xff)])
        }

        // RIFF header
        appendString("RIFF")
        appendUInt32LE(UInt32(36 + dataBytes))
        appendString("WAVE")
        // fmt chunk
        appendString("fmt ")
        appendUInt32LE(16)                              // PCM chunk size
        appendUInt16LE(1)                               // audio format = PCM
        appendUInt16LE(UInt16(numChannels))
        appendUInt32LE(UInt32(sampleRate))
        appendUInt32LE(UInt32(byteRate))
        appendUInt16LE(UInt16(blockAlign))
        appendUInt16LE(UInt16(bitsPerSample))
        // data chunk (all zeroes = silence)
        appendString("data")
        appendUInt32LE(UInt32(dataBytes))
        data.append(Data(count: dataBytes))

        return data
    }
}
