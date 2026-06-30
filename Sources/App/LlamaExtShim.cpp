// LlamaExtShim.cpp
//
// Bridges the llama.cpp "staging" MTP helper functions into a stable C ABI that
// Swift can call through the bridging header (LlamaServer-Bridging-Header.h).
//
// WHY THIS FILE EXISTS
// --------------------
// The MTP helpers we need live in llama.cpp's `llama-ext.h`, which is NOT inside
// the `extern "C"` block of the public `llama.h`. They are therefore exported
// from the dynamic `llama.framework` binary with **C++ mangled** names, e.g.
//
//     llama_set_embeddings_nextn(llama_context*, bool, bool)
//        -> __Z26llama_set_embeddings_nextnP13llama_contextbb
//
// A plain C bridging declaration would emit a reference to the *unmangled*
// symbol `_llama_set_embeddings_nextn` and fail to link. By declaring the same
// prototypes here in C++ (matching argument types exactly) the compiler emits
// the identical mangled symbol, which resolves against the shipped dylib at
// link time. We then re-export thin `extern "C"` wrappers (`cllama_*`) that
// Swift can call with a stable, unmangled ABI.
//
// The context parameter is passed as `void *` across the C boundary so the
// Swift side can hand over its `OpaquePointer` (imported as
// `UnsafeMutableRawPointer`) without a type-identity clash against the
// `llama_context` type already imported from the `llama` module.

#include <cstdint>

// Forward-declare the opaque llama context in the GLOBAL namespace so the
// mangled names match exactly (P13llama_context). We deliberately do NOT include
// llama.h here to keep the mangling under our control and avoid pulling the
// Swift-visible module into C++ translation.
struct llama_context;

// These prototypes MUST match llama.cpp's llama-ext.h signatures exactly so the
// emitted mangled symbols resolve against the shipped framework binary.
//   void  llama_set_embeddings_nextn(llama_context *, bool masked_only, bool ...);
//   float *llama_get_embeddings_nextn(llama_context *);
//   float *llama_get_embeddings_nextn_ith(llama_context *, int32_t i);
//   llama_context *llama_get_ctx_other(llama_context *);
//
// Signatures verified against the stripped dylib symbol table:
//   __Z26llama_set_embeddings_nextnP13llama_contextbb   (ctx, bool, bool)
//   __Z25llama_get_embeddings_nextnP13llama_context     (ctx) -> float*
//   __Z30llama_get_embeddings_nextn_ithP13llama_contexti (ctx, int) -> float*
//   __Z19llama_get_ctx_otherP13llama_context            (ctx) -> llama_context*
void           llama_set_embeddings_nextn(llama_context * ctx, bool enable, bool masked);
float *        llama_get_embeddings_nextn(llama_context * ctx);
float *        llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i);
llama_context *llama_get_ctx_other(llama_context * ctx);

extern "C" {

void cllama_set_embeddings_nextn(void * ctx, bool enable, bool masked) {
    if (ctx == nullptr) { return; }
    llama_set_embeddings_nextn(static_cast<llama_context *>(ctx), enable, masked);
}

const float * cllama_get_embeddings_nextn(void * ctx) {
    if (ctx == nullptr) { return nullptr; }
    return llama_get_embeddings_nextn(static_cast<llama_context *>(ctx));
}

const float * cllama_get_embeddings_nextn_ith(void * ctx, int32_t i) {
    if (ctx == nullptr) { return nullptr; }
    return llama_get_embeddings_nextn_ith(static_cast<llama_context *>(ctx), i);
}

void * cllama_get_ctx_other(void * ctx) {
    if (ctx == nullptr) { return nullptr; }
    return static_cast<void *>(llama_get_ctx_other(static_cast<llama_context *>(ctx)));
}

} // extern "C"
