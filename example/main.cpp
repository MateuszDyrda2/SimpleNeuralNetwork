#include <biai/network.h>
#include <png.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

using namespace biai;
std::vector<float> to_vector(const std::string& path);
int main(int argc, char** argv)
{
    std::vector<dataset::entry_t> entries;

    std::string path = "../data/output3";
    for(auto& entry : std::filesystem::directory_iterator(path))
    {
        auto file = entry.path();
        dataset::entry_t en;
        en.first  = to_vector(file.string());
        auto stem = file.stem();
        if(std::regex_search(stem.string(), std::regex("(square)")))
            en.second = { 1.f, 0.f, 0.f, 0.f };
        else if(std::regex_search(stem.string(), std::regex("(circle)")))
            en.second = { 0.f, 1.f, 0.f, 0.f };
        else if(std::regex_search(stem.string(), std::regex("(triangle)")))
            en.second = { 0.f, 0.f, 1.f, 0.f };
        else if(std::regex_search(stem.string(), std::regex("(star)")))
            en.second = { 0.f, 0.f, 0.f, 1.f };
        else
        {
            std::cerr << "File not matched\n", exit(1);
        }
        entries.push_back(std::move(en));
    }
    size_t inputLayerSize  = entries.front().first.size();
    size_t outputLayerSize = 4;
    size_t hiddenLayerSize = sqrt(inputLayerSize * outputLayerSize);

    auto nt = neural_network::create()
                  .input_layer(inputLayerSize)
                  .hidden_layer(64, neural_network::activation::ReLU)
                  .output_layer(outputLayerSize)
                  .max_epochs(20)
                  .learning_rate(0.01f)
                  .build();
    dataset ds(entries);
    ds.balance();
    std::cout << "Training...\n";
    nt->train(ds);
    std::cout << "Finished\n";

    while(true)
    {
        std::cout << "Pass img to predict";
        std::string path;
        std::cin >> path;
        auto prob = nt->predict(to_vector(path));
        std::cout << "[ ";
        for(auto& elem : prob)
        {
            std::cout << elem << ", ";
        }
        std::cout << "]\n";
    }

    return 0;
}
std::vector<float> to_vector(const std::string& path)
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

    std::vector<float> out(height * width);
    for(size_t y = 0; y < height; ++y)
    {
        auto row = rowPointers[y];
        for(size_t x = 0; x < width; ++x)
        {
            auto px             = row[x];
            float normalized    = px / 255;
            out[y * height + x] = normalized;
        }
    }
    for(int y = 0; y < height; ++y)
    {
        free(rowPointers[y]);
    }
    free(rowPointers);
    return out;
}