#ifndef GAME_H
#define GAME_H
#include <vector>
#include <GL/eglew.h>
#include <GLFW/glfw3.h>

#include "game_level.hpp"
#include "power_up.hpp"

enum GameState {
	GAME_ACTIVE,
	GAME_MENU,
	GAME_WIN
};

enum Direction {
	UP,
	RIGHT,
	DOWN,
	LEFT
};

typedef std::tuple<GLboolean, Direction, glm::vec2> Collision;
const glm::vec2 PLAY_SIZE(100 ,20);
const GLfloat PLAYER_BELOCITY(500.0f);
const glm::vec2 INITIAL_BALL_VELOCITY(100.0f ,-350.0f);
const GLfloat BALL_RADIUS = 12.5f;

class Game {

public:
	GameState States;
	GLboolean Keys[1024];
	GLuint Width, Height;
	std::vector<GameLevel> Levels;
	std::vector<PowerUp> PowerUps;
	GLuint Level;

	Game(GLuint width ,GLuint height);
	~Game();

	void Init();
	void ProcessInput(GLfloat dt);
	void Update(GLfloat dt);
	void Render();
	void DoCollisions(GLfloat dt);

	void ResetLevel();
	void ResetPlayer();

	void SpawnPowerUps(GameObject& block);
	void UpdatePowerUp(GLfloat dt);

};

#endif