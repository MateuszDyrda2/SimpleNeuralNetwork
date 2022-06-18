#include <biai/network.h>
#include <png.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

using namespace biai;
neural_network::vector_t to_vector(const std::string& path);
int main(int argc, char** argv)
{
    std::vector<dataset::entry_t> entries;

    std::string path = "../data/output2";
    std::regex square("(square)");
    std::regex circle("(circle)");
    std::regex triangle("(triangle)");
    for(auto& entry : std::filesystem::directory_iterator(path))
    {
        auto file = entry.path();
        dataset::entry_t en;
        en.first  = to_vector(file.string());
        auto stem = file.stem();
        if(std::regex_search(stem.string(), square))
            en.second = neural_network::vector_t{ { 1.f, 0.f, 0.f } };
        else if(std::regex_search(stem.string(), circle))
            en.second = neural_network::vector_t{ { 0.f, 1.f, 0.f } };
        else if(std::regex_search(stem.string(), triangle))
            en.second = neural_network::vector_t{ { 0.f, 0.f, 1.f } };
        else
        {
            std::cerr << "File not matched\n", exit(1);
        }
        entries.push_back(std::move(en));
    }
    size_t inputLayerSize  = entries.front().first.size();
    size_t outputLayerSize = 3;
    size_t hiddenLayerSize = (inputLayerSize + outputLayerSize) / 2;

    auto nt = neural_network::create()
                  .input_layer(inputLayerSize)
                  .hidden_layer(hiddenLayerSize)
                  .hidden_layer(hiddenLayerSize)
                  .hidden_layer(hiddenLayerSize)
                  .hidden_layer(hiddenLayerSize)
                  .output_layer(outputLayerSize)
                  .build();

    std::cout << "Training...\n";
    nt->train(entries);
    std::cout << "Finished\n";

    while(true)
    {
        std::cout << "Pass img to predict";
        std::string path;
        std::cin >> path;
        Eigen::VectorXf shapeProbability = nt->predict(to_vector(path));
        std::cout << "[ ";
        for(auto& elem : shapeProbability)
        {
            std::cout << elem << ", ";
        }
        std::cout << "]\n";
        // std::cout << shapes->compare(shapeProbability) << std::endl;
    }

    return 0;
}
neural_network::vector_t to_vector(const std::string& path)
{
    FILE* file;
    file      = fopen(path.c_str(), "rb");
    auto png  = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    auto info = png_create_info_struct(png);
    if(setjmp(png_jmpbuf(png)))
        ;
    png_init_io(png, file);
    png_read_info(png, info);
    auto width     = png_get_image_width(png, info);
    auto height    = png_get_image_height(png, info);
    auto colorType = png_get_color_type(png, info);
    auto bitDepth  = png_get_bit_depth(png, info);

    png_read_update_info(png, info);

    auto rowPointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; ++y)
    {
        rowPointers[y] = (png_byte*)malloc(png_get_rowbytes(png, info));
    }
    png_read_image(png, rowPointers);
    fclose(file);
    png_destroy_read_struct(&png, &info, NULL);

    neural_network::vector_t out(height * width);
    for(size_t y = 0; y < height; ++y)
    {
        auto row = rowPointers[y];
        for(size_t x = 0; x < width; ++x)
        {
            auto px             = row[x];
            float normalized    = px / 255;
            out(y * height + x) = normalized;
        }
    }
    for(int y = 0; y < height; ++y)
    {
        free(rowPointers[y]);
    }
    free(rowPointers);
    return out;
}