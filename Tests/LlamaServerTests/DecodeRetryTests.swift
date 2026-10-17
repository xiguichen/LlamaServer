import XCTest

/// Tests for whether a failed generation may be retried transparently.
///
/// Why this exists: a GPU command-buffer fault (`kIOGPUCommandBufferCallbackError
/// InnocentVictim`) wedges the Metal backend, so `llama_decode` fails and the
/// request used to die with a 500 — even when it happened during prefill, before a
/// byte had reached the client. `recoverFromDecodeFailure()` has already rebuilt
/// the context by then, so retrying once is both safe and cheap for the user.
/// The decision is pure, so it is pinned here.
final class DecodeRetryTests: XCTestCase {

    func testRetriesRecoverableDecodeFailureBeforeAnyOutput() {
        XCTAssertTrue(DecodeRetry.shouldRetry(isDecodeFailure: true, retriesSoFar: 0, tokensEmitted: 0),
                      "the context is already rebuilt, nothing was delivered — retrying is invisible to the client")
    }

    func testDoesNotRetryOnceOutputWasDelivered() {
        XCTAssertFalse(DecodeRetry.shouldRetry(isDecodeFailure: true, retriesSoFar: 0, tokensEmitted: 1),
                       "delivered text cannot be taken back, so a retry would duplicate it")
    }

    func testDoesNotRetryMoreThanMaxRetries() {
        XCTAssertFalse(DecodeRetry.shouldRetry(isDecodeFailure: true,
                                               retriesSoFar: DecodeRetry.maxRetries,
                                               tokensEmitted: 0),
                       "a second failure means the backend is not recovering — surface it instead of looping")
    }

    func testDoesNotRetryNonDecodeFailures() {
        XCTAssertFalse(DecodeRetry.shouldRetry(isDecodeFailure: false, retriesSoFar: 0, tokensEmitted: 0),
                       "contextTooFull / tokenize / contextInit are not recoverable by rebuilding the context")
    }
}
