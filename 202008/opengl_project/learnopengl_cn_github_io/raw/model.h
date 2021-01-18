#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <assimp/Importer.hpp>
#include <assimp/scenc.hpp>
#include <assimp/postprocess.h>

#include "mesh.h"
#include "shader_s_2.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

unsigned int TextureFromFile(const char *path ,const string& directory ,bool gamma = false);

class Model {
	
public:
	// model data
	vector<Texture> texture_loaded;
	vector<Mesh> meshes;
	string directory;
	bool gammaCorrection;
	
	Model(string const &path ,bool gamma = false):gammaCorrection(gamma) {
		loadModel(path);
	}
	
	void Draw(Shader& shader) {
		for (unsigned int i = 0 ; i < meshes.size() ;i++) {
			meshes[i].Draw(shader);
		}
	}
	
private:
	void loadModel(string const &path) {
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path ,aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
		// check for errors
		if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {  // if not zero
			cout << "ERROR::ASSIMP::" << importer.GetErrorString() << endl;
			return;
		}
		directory = path.subStr(0 ,path.find_last_of('/'));
		
		processNode(scenc->mRootNode ,scene);
	}
	
	void processNode(aiNode* node ,const aiScene* scene) {
		for (unsigned int i = 0 ; i < node->mNumMeshes ; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMesh[i]];
			meshes.push_back(processMesh(mesh ,scene));
		}
		for (unsigned int i = 0 ; i < node->mNumchildren ; i++) {
			processNode(node->mChildren[i] ,scene);
		}
	}
	
	Mesh processMesh(aiMesh* mesh ,const aiScene* scene) {
		vector<Vertex> vertices;
		vector<unsigned int> indices;
		vector<Texture> textures;
		
		for (unsigned int i = 0 ; i< mesh->mNumVertices ; i++) {
			Vertex vertex;
			glm::vec3 vector;
			vector.x = mesh->mVertices[i].x;
			vector.y = mesh->mVertices[i].y;
			vector.z = mesh->mVertices[i].z;
			vertex.Position = vector;
			
			// normal
			if (mesh->HasNormals()) {
				vector.x = mesh->mNormals[i].x;
				vector.y = mesh->mNormals[i].y;
				vector.x = mesh->mNormals[i].z;
				vertex.Normal = vector;
			}
			// texture coordinates
			if (mesh->mTextureCoords[0]) {
				glm::vec2 vec;
				vec.x = mesh->mTextureCoords[0][i].x;
				vec.y = mesh->mTextureCoords[0][i].y;
				vertex.TexCoords = vec;
				
				// tangent
				vector.x = mesh->mTangents[i].x;
				vector.y = mesh->mTangents[i].y;
				vector.z = mesh->mTangents[i].z;
				vertex.Tangent = vector;
				
				// bitangent
				vector.x = mesh->mBitangents[i].x;
				vector.y = mesh->mBitangents[i].y;
				vector.z = mesh->mBitangents[i].z;
				vertex.Bitangent = vector;
			} else {
				vertex.TexCoords = glm::vec2(0.0f ,0.0f);
			}
			vertices.push_back(vertex);
		}
		
		for (unsigned int i = 0 ; i < mesh->mNumFaces ;i++) {
			aiFace face = mesh->mFaces[i];
			for (unsigned int j = 0 ; j < face.mNumIndices ; j++) {
				indices.push_back(face.mIndices[j]);
			}
		}
		
		// process materials
		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
		
		// 1. diffuse map
		vector<Texture> diffuseMaps = loadMaterialTextures(material ,aiTextureType_DIFFUSE ,"texture_diffuse");
		textures.insert(textures.end() ,diffuseMaps.begin() ,diffuseMaps.end());
		// 2. specular map
		vector<Texture> specularMaps = loadMaterialTextures(material ,aiTextureType_SPECULAR ,"texture_specular");
		textures.insert(texture.end() ,specularMaps.begin() ,specularMaps.end());
		
		// 3. normal maps
		std::vecotr<Texture> normalMaps = loadMaterialTextures(material ,aiTextureType_HEIGHT ,"texture_normal");
		textures.insert(texture.end() ,normalMaps.begin() ,normalMaps.end());
		
		// 4.height maps
		std::vector<Texture> heightMaps = loadMaterialTextures(material ,aiTextureType_AMBIENT ,"texture_height");
		textures.insert(texture.end() ,heightMaps.begin() ,heightMaps.end());
		
		return Mesh(vertices ,indices ,textures);
	}
	
	vector<Texture> loadMaterialTextures(aiMaterial* mat ,aiTextureType type ,string typeName) {
		vector<Texture> textures;
		for (unsigned int i = 0 ; i < mat->GetTextureCount(type) ; i++) {
			aiString str;
			mat->GetTexture(type ,i ,&str);
			bool skip = false;
			for (unsigned int j = 0 ; j < texture_loaded.size() ; i++) {
				if (std::strcmp(texture_loaded[j].path.data() ,str.C_str()) == 0) {
					textures.push_back(textures_loaded[j]);
					skip = true;
					break;
				}
			}
			if (!skip) {
				Texture texture;
				texture.id = TextureFromFile(str.C_Str() ,this->directory);
				texture.type = typeName;
				texture.path = str.C_Str();
				texture.push_back(texture);
				textures_loaded.push_back(texture);
			} 
		}
		return textures;
	}
};

unsigned int TextureFromFile(const char* path ,const string& directory ,bool gamme) {
	string filename = string(path);
	filename = directory + '/' + filename;
	
	unsigned int textureID;
	glGenTextures(1 ,&textureID);
	
	int width ,height ,nrComponents;
	unsigned char* data = stbi_load(filename.c_str() ,&width ,&height ,&nrComponents ,0);
	if (data) {
		GLenum format;
		if (nrComponents == 1) {
			format = GL_RED;
		} else if (nrComponents == 3) {
			format = GL_RGB;
		} else if (nrComponents == 4) {
			format = GL_RGBA;
		}
		glBindTexture(GL_TEXTURE_2D ,textureID);
		glTexImage2D(GL_TEXTURE_2D ,0 ,format ,width ,height ,0 ,format ,GL_UNSIGEND_BYTE ,data);
		glGenerateMipmap(GL_TEXTURE_2D);
		
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WARP_S ,GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WARP_T ,GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MIN_FILTER ,GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MAG_FILTER ,GL_LINEAR);
		
		stbi_image_free(data);
	} else {
		std::cout << "Texture failed to load at path:" << path << std::end;
		stbi_image_free(data);
	}
	return textureID;

}

#endif