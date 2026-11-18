#pragma once
// Process-wide store of images that live inside the project instead of on
// disk: AI generations and Versions snapshots. The Python side owns the
// encoded bytes (saved in the .vcp) and registers the decoded pixels here
// under a content id; ops reference them through a short string knob, so
// undo snapshots and node hashes never carry the pixel data.

#include <cstdint>
#include <string>
#include <vector>

#include "vc/image.h"
#include "vc/util.h"

namespace vc::ops {

// Register (or replace) an image. The image is shared, never copied on read.
void setImageBlob(const std::string& id, ImagePtr img);
// nullptr when the id is unknown.
ImagePtr imageBlob(const std::string& id);
// Changes whenever the blob behind `id` is (re)registered or removed; 0 when absent.
uint64_t imageBlobSerial(const std::string& id);
bool removeImageBlob(const std::string& id);
void clearImageBlobs();
std::vector<std::string> imageBlobIds();
size_t imageBlobBytes();

}  // namespace vc::ops
