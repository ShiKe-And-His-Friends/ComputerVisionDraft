#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <map>
#include <string>

#include <GL/eglew.h>

#include <texture.h>
#include <shader.h>

class ResourceManager {

public:
	static std::map<std::string, Shader> Shaders;
	static std::map<std::string, Texture2D> Textures;
	static Shader LoadShader(const GLchar *vShaderFile ,const GLchar* fShaderFile ,const GLchar *gShaderFile ,std::string name);
};

#endif 