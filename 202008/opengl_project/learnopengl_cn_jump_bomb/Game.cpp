#include "Game.hpp"
#include "resource_manager.hpp"
#include "sprite_renderer.hpp"
#include <iostream>

SpriteRenderer* Renderer;

Game::Game(GLuint width ,GLuint height):State(GAME_ACTIVE) ,Keys() ,Width(width) ,Height(height) {

}

Game::~Game() {

}

void Game::Init() {
	fprintf(stderr , "shikeDebug 111");
	ResourceManager::LoadShader("sprite.vs" ,"sprite.frag" ,nullptr ,"sprite");
	glm::mat4 projection = glm::ortho(0.0f ,static_cast<GLfloat>(this->Width),
		static_cast<GLfloat>(this->Height) ,0.0f ,-1.0f ,1.0f);
	ResourceManager::GetShader("sprite").Use().SetInteger("image" ,0);
	ResourceManager::GetShader("sprite").SetMatrix4("projection" ,projection);
	Renderer = new SpriteRenderer(ResourceManager::GetShader("sprite"));
	ResourceManager::LoadTexture("D:\SVNCodeWorkSpace\shikeGitTemp\ComputerVisionDraft\ComputerVisionDraft\202008\opengl_project\learnopengl_cn_github_io\drawable\awesomeface.png" ,GL_TRUE ,"face");
	fprintf(stderr, "shikeDebug 222");
}

void Game::Update(GLfloat dt) {

}

void Game::ProcessInput(GLfloat dt) {

}

void Game::Render() {
	Renderer->DrawSprite(ResourceManager::GetTexture("face"),
		glm::vec2(200, 200) ,glm::vec2(300 ,400) ,45.0f ,glm::vec3(0.0f ,1.0f ,0.0f));
}