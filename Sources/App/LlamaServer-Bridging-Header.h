// LlamaServer-Bridging-Header.h
//
// Objective-C / C bridging header exposing the C++ MTP shim wrappers
// (implemented in LlamaExtShim.cpp) to Swift. See LlamaExtShim.cpp for why a
// shim is needed (the underlying llama.cpp helpers are C++-mangled).
//
// The context is passed as `void *` so Swift can hand over an OpaquePointer
// (bridged as UnsafeMutableRawPointer) without a type clash against the
// `llama_context` type imported from the `llama` module.

#ifndef LLAMASERVER_BRIDGING_HEADER_H
#define LLAMASERVER_BRIDGING_HEADER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Enable/disable emission of the post-norm next-token hidden state on a context.
/// On the TARGET context call with masked=false (dense per-token hidden states).
/// On the MTP/draft context call with masked=true.
void cllama_set_embeddings_nextn(void *ctx, bool enable, bool masked);

/// Pointer to the next-token hidden state for the last decoded output row.
const float *cllama_get_embeddings_nextn(void *ctx);

/// Pointer to the next-token hidden state for batch/output row `i`.
const float *cllama_get_embeddings_nextn_ith(void *ctx, int32_t i);

/// The "other" context associated with this one (target<->draft linkage).
void *cllama_get_ctx_other(void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* LLAMASERVER_BRIDGING_HEADER_H */
