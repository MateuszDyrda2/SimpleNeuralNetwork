#include <biai/network.h>
#include <png.h>

#include <Eigen/Dense>

#include <iostream>

using namespace biai;

class PNG
{
  public:
    bool readPngFileAsGrayScale(std::string filename);
    Eigen::MatrixXf networkInputLayer();

  private:
    int width, height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep* row_pointers = nullptr;
};

int main(int argc, char** argv)
{
    PNG png;
    png.readPngFileAsGrayScale(argv[1]);
    std::cout << png.networkInputLayer();
    return 0;
}

Eigen::MatrixXf PNG::networkInputLayer()
{
    Eigen::MatrixXf m(height, width);
    for(int y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++)
        {
            png_bytep px      = &(row[x * 2]);
            float grayX = px[0]; 
            float normalizedGrayX = grayX / 255; // make range 0..255 to 0..1
            m(y, x) = normalizedGrayX;
        }
    }
    return m;
}

bool PNG::readPngFileAsGrayScale(std::string filename)
{
     FILE* file;
     fopen_s(&file, filename.c_str(), "rb");

     png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png)
        return false;

    png_infop info = png_create_info_struct(png);
    if(!info)
        return false;

    if(setjmp(png_jmpbuf(png)))
        return false;

    png_init_io(png, file);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth  = png_get_bit_depth(png, info);

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // Fill alpha channel
    if(color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    // CONVERT TO GRAY SCALE
    //
    //  error_action = 1: silently do the conversion
    //     error_action = 2: issue a warning if the original
    //                       image has any pixel where
    //                       red != green or red != blue
    //     error_action = 3: issue an error and abort the
    //                       conversion if the original
    //                       image has any pixel where
    //                       red != green or red != blue
    //
    //     red_weight:       weight of red component times 100000
    //     green_weight:     weight of green component times 100000
    //                       If either weight is negative, default
    //                       weights (21268, 71514) are used.
    // http://www.libpng.org/pub/png/libpng-1.2.5-manual.html
     if(color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA)
     {
        png_set_rgb_to_gray_fixed(png, 1, 21268, 71514);
     }
        
    else if(color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(png);
        png_set_rgb_to_gray_fixed(png, 2, 21268, 71514);
    }
    png_read_update_info(png, info);

    if(row_pointers) abort();

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++)
    {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);

    fclose(file);

    png_destroy_read_struct(&png, &info, NULL);

    return true;
}