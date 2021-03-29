#include "resource_manager.hpp"
#include <iostream>
#include <sstream>
#include <fstream>

#include <SOIL.h>

std::map<std::string, Texture2D> ResourceManager::Textures;
std::map<std::string, Shader> ResourceManager::Shaders;