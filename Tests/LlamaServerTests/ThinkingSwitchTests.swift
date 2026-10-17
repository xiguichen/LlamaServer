import XCTest

/// `ThinkingSwitch` is the only place `enable_thinking` reaches a prompt: the
/// vendored `llama_chat_apply_template` takes no template kwargs, so the model's
/// own template always renders the *thinking on* tail (verified on device — the
/// inference log shows the same bare `<|im_start|>assistant\n` for both
/// `enable_thinking` values).
final class ThinkingSwitchTests: XCTestCase {

    // MARK: - thinking off

    /// The model's template and `LlamaInference.fallbackPrompt` both end at the
    /// bare primer, so the block has to be opened *and* closed.
    func testClosesBarePrimer() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertEqual(ThinkingSwitch.apply(to: prompt, thinkingEnabled: false),
                       prompt + "<think>\n\n</think>\n\n")
    }

    /// A renderer that already opened the block is missing only the closer.
    func testClosesOpenPrimer() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        XCTAssertEqual(ThinkingSwitch.apply(to: prompt, thinkingEnabled: false),
                       prompt + "\n</think>\n\n")
    }

    /// Any other tail is left alone: there is no block to close, and adding one
    /// would corrupt a prompt that is already correct.
    func testLeavesUnknownTailAlone() {
        let prompt = "<|im_start|>user\nhi<|im_end|>\n"
        XCTAssertEqual(ThinkingSwitch.apply(to: prompt, thinkingEnabled: false), prompt)
    }

    /// Idempotent: after it is applied the tail is no longer a primer, so a
    /// second pass changes nothing.
    func testApplyingTwiceChangesNothingMore() {
        let prompt = "<|im_start|>assistant\n"
        let once = ThinkingSwitch.apply(to: prompt, thinkingEnabled: false)
        XCTAssertEqual(ThinkingSwitch.apply(to: once, thinkingEnabled: false), once)
    }

    // MARK: - thinking on

    /// Opting in must never touch the prompt.
    func testThinkingOnIsUntouched() {
        for prompt in ["<|im_start|>assistant\n",
                       "<|im_start|>assistant\n<think>\n",
                       "<|im_start|>user\nhi<|im_end|>\n"] {
            XCTAssertEqual(ThinkingSwitch.apply(to: prompt, thinkingEnabled: true), prompt)
        }
    }
}
