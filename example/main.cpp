#include <biai/network.h>
#include <png.h>

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <map>
#include <string>

class PNG
{
  public:
    PNG(std::string filename);
    bool readPngFileAsGrayScale(std::string filename);
    Eigen::VectorXf networkInputLayer();

  private:
    int width, height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep* row_pointers = nullptr;
};

class Shapes
{
    public:
        Shapes();
        Eigen::VectorXf getVec(std::string shape);
        std::string compare(Eigen::VectorXf probability);
    private:
        std::map<std::string, Eigen::VectorXf> map;
};

//First argument should be folder with training data
//Second argument should be png to guess its shape
int main(int argc, char** argv)
{
    std::unique_ptr<Shapes> shapes = std::make_unique<Shapes>();
    //load every training png from data folder as grayscale
    std::vector<Eigen::VectorXf> inputs;
    std::vector<Eigen::VectorXf> outputs;
    std::string path = argv[1];
    for (auto & entry : fs::directory_iterator(path))
    {
        inputs.push_back(PNG(entry).networkInputLayer());

        std::string shape = entry.substr(0, entry.find("_"));
        outputs.push_back(shapes->getVec(shape));   
    }

    
    //number of training pngs
    std::size_t input_layer_size = inputs.size();
    //number of shapes to guess
    std::size_t ouput_layer_size = 9;
    //idk
    std::size_t hidden_layer_size = 69;

    //create neural network
    std::unique_ptr<biai::neural_network> nt = biai::neural_network::create()
    .input_layer(input_layer_size)
    .output_layer(ouput_layer_size)
    .hidden_layer(hidden_layer_size)
    .activation_function(std::make_unique<biai::sigmoid>())
    .build();

    //train neural network
    nt->evaluate(inputs, outputs);

    //guees the shape
    Eigen::VectorXf shapeProbability = nt->predict(PNG(argv[2]).networkInputLayer());
    std::cout << shapes->compare(shapeProbability) << std::endl;

    return 0;
}

PNG::PNG(std::string filename)
{
    readPngFileAsGrayScale(filename);
}

Eigen::VectorXf PNG::networkInputLayer()
{
    Eigen::VectorXf vec(height * width);
    for(int y = 0; y < height; y++)
    {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < width; x++)
        {
            png_bytep px      = &(row[x * 2]);
            float grayX = px[0]; 
            float normalizedGrayX = grayX / 255; // make range 0..255 to 0..1
            vec(y * height + x) = normalizedGrayX;
        }
    }
    return vec;
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

Shapes::Shapes()
{
    map.insert({"Circle",   Eigen::VectorXf{1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}});
    map.insert({"Hexagon",  Eigen::VectorXf{0.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}});
    map.insert({"Heptagon", Eigen::VectorXf{0.f,0.f,1.f,0.f,0.f,0.f,0.f,0.f,0.f}});
    map.insert({"Nonagon",  Eigen::VectorXf{0.f,0.f,0.f,1.f,0.f,0.f,0.f,0.f,0.f}});
    map.insert({"Octagon",  Eigen::VectorXf{0.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,0.f}});
    map.insert({"Pentagon", Eigen::VectorXf{0.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f}});
    map.insert({"Square",   Eigen::VectorXf{0.f,0.f,0.f,0.f,0.f,0.f,1.f,0.f,0.f}});
    map.insert({"Star",     Eigen::VectorXf{0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f,0.f}});
    map.insert({"Triangle", Eigen::VectorXf{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,1.f}});
}

Eigen::VectorXf Shapes::getVec(std::string shape)
{
   return map.at(shape);
}

std::string Shapes::compare(Eigen::VectorXf probability)
{
    float highestValue = -1.f;
    for (size_t  i = 0; i < probability.size(); i++)
    {
        if (probability[i] > highestValue)
        {
            highestValue = probability[i];
        }
    }
    Eigen::VectorXf result(9);
    for (size_t  i = 0; i < probability.size(); i++)
    {
        if (probability[i] == highestValue)
        {
            result << 1.f;
        }
        else
        {
            result << 0.f;
        }
    }
    for (const auto& kv : map) 
    {
        if (kv.second == result)
            return kv.first;
    }
    return "error";
}