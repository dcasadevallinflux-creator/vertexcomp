/* stb implementations compiled as C so tinyexr's extern "C" zlib declarations link. */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"
