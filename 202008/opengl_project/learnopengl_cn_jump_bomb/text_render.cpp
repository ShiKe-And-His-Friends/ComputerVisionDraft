#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

#include "text_render.hpp"
#include "resource_manager.hpp"

TextRenderer::TextRenderer(GLuint width ,GLuint height){
	this->TextShader = ResourceManager::LoadShader("text_draw.vs" ,"text_draw.frag" ,nullptr ,"text");
	this->TextShader.SetMatrix4("projection" ,glm::ortho(0.0f ,static_cast<GLfloat>(width) ,static_cast<GLfloat>(height) ,0.0f),GL_TRUE);
	this->TextShader.SetInteger("text",0);
	glGenVertexArrays(1 ,&this->VAO);
	glGenBuffers(1 ,&this->VBO);
	glBindVertexArray(this->VAO);
	glBindBuffer(GL_ARRAY_BUFFER ,this->VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(GLfloat) * 6 *4 ,NULL ,GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0 ,4 ,GL_FLOAT ,GL_FALSE ,4 * sizeof(GLfloat) ,0);
	glBindBuffer(GL_ARRAY_BUFFER ,0);
	glBindVertexArray(0);
}

void TextRenderer::Load(std::string font ,GLuint fontSize) {
	this->Characters.clear();
	FT_Library ft;
	if (FT_Init_FreeType(&ft)) {
		std::cout << "ERROR::FREETYPE Could not init FreeType Library" << std::endl;
	}
	FT_Face face;
	if (FT_New_Face(ft, font.c_str(), 0, &face)) {
		std::cout << "ERROR::FREETYPE Failed to load font" << std::endl;
	}
	FT_Set_Pixel_Sizes(face ,0 ,fontSize);
	glPixelStorei(GL_UNPACK_ALIGNMENT ,1);
	for (GLubyte c = 0; c < 128; c++) {
		if (FT_Load_Char(face ,c ,FT_LOAD_RENDER)) {
			std::cout << "ERROR::FREETYPE Failed to load GLyph" << std::endl;
			continue;
		}
		GLuint texture;
		glGenTextures(1 ,&texture);
		glBindTexture(GL_TEXTURE_2D ,texture);
		glTexImage2D(
			GL_TEXTURE_2D
			,0
			,GL_RED
			,face->glyph->bitmap.width
			,face->glyph->bitmap.rows
			,0
			,GL_RED
			,GL_UNSIGNED_BYTE
			,face->glyph->bitmap.buffer);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_S ,GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MIN_FILTER ,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		Character characters = {
			texture
			,glm::ivec2(face->glyph->bitmap.width ,face->glyph->bitmap.rows)
			,glm::ivec2(face->glyph->bitmap_left ,face->glyph->bitmap_top)
			,face->glyph->advance.x
		};
		Characters.insert(std::pair<GLchar ,Character>(c , characters));
	}
	glBindTexture(GL_TEXTURE_2D ,0);
	FT_Done_Face(face);
	FT_Done_FreeType(ft);
}

void TextRenderer::RenderText(std::string text ,GLfloat x ,GLfloat y ,GLfloat scale ,glm::vec3 color) {
	this->TextShader.Use();
	this->TextShader.SetVector3f("textColor" ,color);
	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(this->VAO);

	std::string::const_iterator c;
	for (c = text.begin(); c != text.end(); c++) {
		Character ch = Characters[*c];
		GLfloat xpos = x + ch.Breaking.x * scale;
		GLfloat ypos = y + (this->Characters['H'].Breaking.y - ch.Breaking.y) * scale;
		GLfloat w = ch.Size.x * scale;
		GLfloat h = ch.Size.y * scale;
		GLfloat vertices[6][4] = {
			{xpos	,ypos+h	,0.0	,1.0},
			{xpos+w	,ypos	,1.0	,0.0},
			{xpos	,ypos	,0.0	,0.0},

			{xpos	,ypos+h	,0.0	,1.0},
			{xpos+w	,ypos+h	,1.0	,1.0},
			{xpos+w	,ypos	,1.0	,0.0}
		};
		glBindTexture(GL_TEXTURE_2D ,ch.TextureID);
		glBindBuffer(GL_ARRAY_BUFFER ,this->VBO);
		glBufferSubData(GL_ARRAY_BUFFER ,0 ,sizeof(vertices),vertices);
		glBindBuffer(GL_ARRAY_BUFFER ,0);
		glDrawArrays(GL_TRIANGLES ,0 ,6);
		x += (ch.Advance >> 6) * scale;
	}
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D ,0);
}