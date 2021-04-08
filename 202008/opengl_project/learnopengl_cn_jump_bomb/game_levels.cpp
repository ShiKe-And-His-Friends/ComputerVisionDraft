#include "game_levels.hpp"
#include "resource_manager.hpp"
#include "sprite_renderer.hpp"
#include "game_object.hpp"
#include "ball_object_collision.hpp"
#include "particle_generator.hpp"

SpriteRenderer* Renderer;
GameObject* Player;
BallObject* Ball;
ParticleGenerator* Particles;

Game::Game(GLuint width ,GLuint height) : States(GAME_ACTIVE) ,Keys() ,Width(width) ,Height(height){
	this->Width = width;
	this->Height = height;
}

Game::~Game() {
	delete Renderer;
	delete Player;
	delete Ball;
	delete Particles;
}

void Game::Init() {

	ResourceManager::LoadShader("sprite.vs" ,"sprite.frag" ,nullptr ,"sprite");
	ResourceManager::LoadShader("particle.vs", "particle.vs", nullptr, "particle");
	glm::mat4 project = glm::ortho(0.0f ,static_cast<GLfloat>(this->Width) ,static_cast<GLfloat>(this->Height) ,0.0f ,-1.0f ,1.0f);
	ResourceManager::GetShader("sprite").Use().SetInteger("sprite" ,0);
	ResourceManager::GetShader("sprite").SetMatrix4("projection" ,project);
	ResourceManager::GetShader("particle").Use().SetInteger("sprite", 0);
	ResourceManager::GetShader("particle").SetMatrix4("projection", project);

	ResourceManager::LoadTexture("background.jpg" ,GL_FALSE ,"background");
	ResourceManager::LoadTexture("../learnopengl_cn_github_io/drawable/awesomeface.png" ,GL_TRUE ,"face");
	ResourceManager::LoadTexture("block.png", GL_FALSE, "block");
	ResourceManager::LoadTexture("block_solid.png", GL_FALSE, "block_solid");
	ResourceManager::LoadTexture("paddle.png", GL_TRUE, "paddle");
	ResourceManager::LoadTexture("particle.png", GL_TRUE, "particle");

	Renderer = new SpriteRenderer(ResourceManager::GetShader("sprite"));
	Particles = new ParticleGenerator(ResourceManager::GetShader("particle") , ResourceManager::GetTexture("particle") ,500);

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

	glm::vec2 ballPos = playerPos + glm::vec2(PLAY_SIZE.x /2 - BALL_RADIUS ,-BALL_RADIUS * 2);
	Ball = new BallObject(ballPos ,BALL_RADIUS ,INITIAL_BALL_VELOCITY ,ResourceManager::GetTexture("face"));
}

void Game::Update(GLfloat dt) {
	Ball->Move(dt ,this->Width);
	this->DoCollisions(dt);
	Particles->Update(dt ,*Ball ,2 ,glm::vec2(Ball->Radius /2));
	if (Ball->Position.y >= this->Height) {
		this->ResetLevel();
		this->ResetPlayer();
	}
}

void Game::ProcessInput(GLfloat dt) {
	if (this->States == GAME_ACTIVE) {
		GLfloat velocity = PLAYER_BELOCITY * dt;
		if (this->Keys[GLFW_KEY_A]) {
			//std::cout << "a x=" << Player->Position.x << std::endl;
			if (Player->Position.x >= 0) {
				Player->Position.x -= velocity;
				if (Ball->Stuck) {
					Ball->Position.x -= velocity;
				}
			}
		}

		if (this->Keys[GLFW_KEY_D]) {
			//std::cout << "w x=" << Player->Position.x << " screen=" << this->Width << " player=" << Player->Size.x << std::endl;
			if (Player->Position.x <= this->Width - Player->Size.x) {
				Player->Position.x += velocity;
				if (Ball->Stuck) {
					Ball->Position.x += velocity;
				}
			}
		}
		if (this->Keys[GLFW_KEY_SPACE]) {
			Ball->Stuck = false;
		}
	}
}

void Game::Render() {
	if (this->States == GAME_ACTIVE) {
		Renderer->DrawSprite(ResourceManager::GetTexture("background") ,glm::vec2(0 ,0) ,glm::vec2(this->Width ,this->Height) ,0.0f);
		this->Levels[this->Level].Draw(*Renderer);
		Player->Draw(*Renderer);
		Particles->Draw();
		Ball->Draw(*Renderer);
	}
}

void Game::ResetLevel() {
	if (this->Level == 0) {
		this->Levels[0].load("one.lvl", this->Width, this->Height * 0.5);
	}
	else if (this->Level == 1) {
		this->Levels[1].load("two.lvl", this->Width, this->Height * 0.5);
	}
	else if (this->Level == 2) {
		this->Levels[2].load("three.lvl", this->Width, this->Height * 0.5);
	}
	else if (this->Level == 3) {
		this->Levels[3].load("four.lvl", this->Width, this->Height * 0.5);
	}

}

void Game::ResetPlayer() {
	Player->Size = PLAY_SIZE;
	Player->Position = glm::vec2(this->Width /2 - PLAY_SIZE.x ,this->Height - PLAY_SIZE.y);
	Ball->Reset(Player->Position + glm::vec2(PLAY_SIZE .x/2 - BALL_RADIUS ,-(BALL_RADIUS * 2)) , INITIAL_BALL_VELOCITY);
}

GLboolean CheckCollision(GameObject &one ,GameObject &two);
Collision CheckCollision(BallObject& one, GameObject& two);
Direction VectorDirection(glm::vec2 closet);

void Game::DoCollisions(GLfloat dt) {
	for (GameObject& box : this->Levels[this->Level].Bricks) {
		if (!box.Destroyed) {
			Collision collision = CheckCollision(*Ball ,box);
			if (std::get<0>(collision)) {
				if (!box.IsSolid) {
					box.Destroyed = GL_TRUE;
				}
				Direction dir = std::get<1>(collision);
				glm::vec2 diff_vector = std::get<2>(collision);
				if (dir == LEFT || dir == RIGHT) {
					Ball->Velocity.x = -Ball->Velocity.x;
					GLfloat penetraion = Ball->Radius - std::abs(diff_vector.x);
					if (dir == LEFT) {
						Ball->Position.x += penetraion;
					}
					else {
						Ball->Position.x -= penetraion;
					}
				}
				else {
					Ball->Velocity.y = -Ball->Velocity.y;
					GLfloat penetration = Ball->Radius - std::abs(diff_vector.y);
					if (dir == UP) {
						Ball->Position.y -= penetration;
					}
					else {
						Ball->Position.y += penetration;
					}
				}
			}

		}
	}
	Collision result = CheckCollision(*Ball ,*Player);
	if (!Ball->Stuck && std::get<0>(result)) {
		GLfloat centerBoard = Player->Position.x + Player->Size.x / 2;
		GLfloat distance = (Ball->Position.x + Ball->Radius) - centerBoard;
		GLfloat percemtage = distance / (Player->Size.x /2);
		GLfloat strengeth = 2.0f;
		glm::vec2 oldVelocity = Ball->Velocity;
		Ball->Velocity.x = INITIAL_BALL_VELOCITY.x * percemtage * strengeth;
		Ball->Velocity = glm::normalize(Ball->Velocity) * glm::length(oldVelocity);
		Ball->Velocity.y = -1 * abs(Ball->Velocity.y);
	}

}

GLboolean CheckCollision(GameObject &one ,GameObject &two) {
	bool collisionX = one.Position.x + one.Size.x >= two.Position.x
		&&two.Position.x + two.Size.x >= one.Position.x;
	bool collisionY = one.Position.y + one.Size.y >= two.Position.y
		&& two.Position.y + two.Size.y >= one.Position.y;
	return collisionX && collisionY;
}

Collision CheckCollision(BallObject& one, GameObject& two) {
	glm::vec2 center(one.Position  + one.Radius);
	glm::vec2 aabb_half_extents(two.Size.x /2 ,two.Size.y /2);
	glm::vec2 aabb_center(two.Position.x + aabb_half_extents.x 
			,two.Position.y + aabb_half_extents.y);
	glm::vec2 difference = center - aabb_center;
	glm::vec2 clamped = glm::clamp(difference ,-aabb_half_extents ,aabb_half_extents);
	glm::vec2 closest = aabb_center + clamped;
	difference = closest - center;
	if (glm::length(difference) < one.Radius) {
		return std::make_tuple(GL_TRUE ,VectorDirection(difference) ,difference);
	}
	else {
		return std::make_tuple(GL_FALSE ,UP ,glm::vec2(0 ,0));
	}
}

Direction VectorDirection(glm::vec2 target) {
	glm::vec2 compass[] = {
		glm::vec2(0.0f ,1.0f),
		glm::vec2(1.0f ,1.0f),
		glm::vec2(0.0f ,-1.0f),
		glm::vec2(-1.0f ,0.0f)
	};
	GLfloat max = 0.0f;
	GLuint best_mathc = -1;
	for (GLuint i = 0; i < 4;i++) {
		GLfloat dot_product = glm::dot(glm::normalize(target) ,compass[i]);
		if (dot_product > max) {
			max = dot_product;
			best_mathc = i;
		}
	}
	return (Direction)best_mathc;
}