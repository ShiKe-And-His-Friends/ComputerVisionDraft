#include "game_setting_up.h"
#include "resource_manager.h"

Game::Game(Gluint width ,GLuint height) 
	: State(GAME_ACTIVE) .Keys() ,Width(width) ,Height(height){
		
}

Game::~Game() {

}

void Game::Init() {

}

void Game::Update(GLfloat dt) {

}

void Game::ProcessInput(GLfloat dt) {

}

void Game::Render() {

}
