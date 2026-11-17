#pragma once
#include "vc/graph.h"

namespace vc::ops {

// Render a Write node's input for `frame`, converted to the node's output
// colourspace and alpha convention. Used for image files and by the host
// movie encoder.
MutImagePtr prepareWriteImage(Engine& engine, const NodePtr& node, double frame, ChannelList& chans, Format& fmt,
                              const CancelToken* cancel);

// Write one frame of an image-file Write node. Returns false when skipped.
bool writeFrame(Engine& engine, const NodePtr& node, double frame, const CancelToken* cancel);

}  // namespace vc::ops
