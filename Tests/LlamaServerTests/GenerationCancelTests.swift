import XCTest

/// Why an in-flight generation stops at its next safe boundary.
///
/// The 2026-11-15 device log showed a client that closed the socket 1.5 s into a
/// 4,451-token prefill and the server kept decoding for the full 20 s, because
/// the prompt loop only knew about server shutdown. These tests pin the decision
/// (and the precedence) that both paths now share.
final class GenerationCancelTests: XCTestCase {

    func testNoCancelWhileClientIsConnected() {
        XCTAssertNil(GenerationCancel.cause(isShuttingDown: false,
                                            isClientDisconnected: false))
    }

    func testClientDisconnectCancelsPrefill() {
        XCTAssertEqual(GenerationCancel.cause(isShuttingDown: false,
                                              isClientDisconnected: true),
                       .clientDisconnected)
    }

    func testShutdownWinsWhenBothAreTrue() {
        XCTAssertEqual(GenerationCancel.cause(isShuttingDown: true,
                                              isClientDisconnected: true),
                       .serverStopping,
                       "stop() drains the queue and its breadcrumbs expect the shutdown path")
    }

    func testCauseLogDescription() {
        XCTAssertEqual(GenerationCancel.serverStopping.logDescription, "server stopping")
        XCTAssertEqual(GenerationCancel.clientDisconnected.logDescription, "client disconnected")
    }
}
