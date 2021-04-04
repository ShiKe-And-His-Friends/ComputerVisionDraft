#include "game_levels.hpp"
#include "resource_manager.hpp"
#include "sprite_renderer.hpp"
#include "game_object.hpp"

SpriteRenderer* Renderer;
GameObject* Player;

Game::Game(GLuint width ,GLuint height) : States(GAME_ACTIVE) ,Keys() ,Width(width) ,Height(height){
	
}

Game::~Game() {
	delete Renderer;
	delete Player;
}

void Game::Init() {

	ResourceManager::LoadShader("sprite.vs" ,"sprite.frag" ,nullptr ,"sprite");
	glm::mat4 project = glm::ortho(0.0f ,static_cast<GLfloat>(this->Width) ,static_cast<GLfloat>(this->Height) ,0.0f ,-1.0f ,1.0f);
	ResourceManager::GetShader("sprite").Use().SetInteger("sprite" ,0);
	ResourceManager::GetShader("sprite").SetMatrix4("projection" ,project);

	ResourceManager::LoadTexture("background.jpg" ,GL_FALSE ,"background");
	ResourceManager::LoadTexture("awesomeface.png" ,GL_TRUE ,"face");
	ResourceManager::LoadTexture("block.png", GL_FALSE, "block");
	ResourceManager::LoadTexture("block_solid.png", GL_FALSE, "block_solid");
	ResourceManager::LoadTexture("paddle.png", true, "paddle");

	Renderer = new SpriteRenderer(ResourceManager::GetShader("sprite"));

	GameLevel one, two, three, four;
	one.load("one.lvl" ,this->Width ,this->Height * 0.5);
	two.load("two.lvl", this->Width, this->Height * 0.5);
	three.load("three.lvl", this->Width, this->Height * 0.5);
	four.load("four.lvl", this->Width, this->Height * 0.5);
	this->Levels.push_back(one);
	this->Levels.push_back(two);
	this->Levels.push_back(three);
	this->Levels.push_back(four);
	this->Level = 0;
	glm::vec2 playerPos = glm::vec2(this->Width /2 - PLAY_SIZE.x /2 ,this->Height - PLAY_SIZE.y);
	Player = new GameObject(playerPos ,PLAY_SIZE ,ResourceManager::GetTexture("paddle"));
}

void Game::Update(GLfloat dt) {
	
}

void Game::ProcessInput(GLfloat dt) {
	if (this->States == GAME_ACTIVE) {
		GLfloat velocity = PLAYER_BELOCITY * dt;
		if (this->Keys[GLFW_KEY_A]) {
			if (Player->Position.x >= 0) {
				Player->Position.x -= velocity;
			}
		}

		if (this->Keys[GLFW_KEY_D]) {
			if (Player->Position.x <= this->Width - Player->Size.x) {
				Player->Position.x += velocity;
			}
		}
	}
}

void Game::Render() {
	if (this->States == GAME_ACTIVE) {
		Renderer->DrawSprite(ResourceManager::GetTexture("background") ,glm::vec2(0 ,0) ,glm::vec2(this->Width ,this->Height) ,0.0f);
		this->Levels[this->Level].Draw(*Renderer);
		Player->Draw(*Renderer);
	}
}