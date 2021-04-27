#include <sstream>
#include "game_levels.hpp"
#include "resource_manager.hpp"
#include "sprite_renderer.hpp"
#include "game_object.hpp"
#include "ball_object_collision.hpp"
#include "particle_generator.hpp"
#include "post_process.hpp"
#include "text_render.hpp"

SpriteRenderer* Renderer;
GameObject* Player;
BallObject* Ball;
ParticleGenerator* Particles;
PostProcessor* Effects;
GLfloat ShakeTime = 0.0f;

using namespace std;
using namespace irrklang;

ISoundEngine* SoundEngine = createIrrKlangDevice();
TextRenderer* Text;

Game::Game(GLuint width ,GLuint height) : States(GAME_MENU) ,Keys() ,Width(width) ,Height(height) ,Level(0) ,Lives(3){
	this->Width = width;
	this->Height = height;
}

Game::~Game() {
	delete Renderer;
	delete Player;
	delete Ball;
	delete Particles;
	delete Effects;
	delete Text;
	SoundEngine->drop();
}

void Game::Init() {

	ResourceManager::LoadShader("sprite.vs" ,"sprite.frag" ,nullptr ,"sprite");
	ResourceManager::LoadShader("particle.vs", "particle.frag", nullptr, "particle");
	ResourceManager::LoadShader("post_process.vs", "post_process.frag", nullptr, "postprocessing");
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
	ResourceManager::LoadTexture("powerup_speed.png", GL_TRUE, "powerup_speed");
	ResourceManager::LoadTexture("powerup_sticky.png", GL_TRUE, "powerup_sticky");
	ResourceManager::LoadTexture("powerup_increase.png", GL_TRUE, "powerup_increase");
	ResourceManager::LoadTexture("powerup_confuse.png", GL_TRUE, "powerup_confuse");
	ResourceManager::LoadTexture("powerup_chaos.png", GL_TRUE, "powerup_chaos");
	ResourceManager::LoadTexture("powerup_passthrough.png", GL_TRUE, "powerup_passthrough");

	Renderer = new SpriteRenderer(ResourceManager::GetShader("sprite"));
	Particles = new ParticleGenerator(ResourceManager::GetShader("particle") , ResourceManager::GetTexture("particle") ,500);
	Effects = new PostProcessor(ResourceManager::GetShader("postprocessing"), this->Width, this->Height);
	Text = new TextRenderer(this->Width ,this->Height);
	Text->Load("arial.ttf" ,24);

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
	SoundEngine->play2D("irrklang/breakout.mp3" ,GL_TRUE);
}

void Game::Update(GLfloat dt) {
	Ball->Move(dt ,this->Width);
	this->DoCollisions(dt);
	Particles->Update(dt ,*Ball ,2 ,glm::vec2(Ball->Radius /2));
	this->UpdatePowerUp(dt);
	if (ShakeTime > 0.0f) {
		ShakeTime -= dt;
		if (ShakeTime <= 0.0f) {
			Effects->Shake = GL_FALSE;
		}
	}
	if (Ball->Position.y >= this->Height) {
		--this->Lives;
		if (this->Lives == 0) {
			this->ResetLevel();
			this->States == GAME_MENU;
		}
		this->ResetPlayer();
	}
	if (this->States == GAME_ACTIVE && this->Levels[this->Level].IsCompleted()) {
		this->ResetLevel();
		this->ResetPlayer();
		Effects->Chaos = GL_TRUE;
		this->States = GAME_WIN;
	}
}

void Game::ProcessInput(GLfloat dt) {
	if (this->States == GAME_MENU) {
		if (this->States == GAME_MENU) {
			if (this->Keys[GLFW_KEY_ENTER] && !this->KeysProcessed[GLFW_KEY_ENTER]) {
				this->States = GAME_ACTIVE;
				this->KeysProcessed[GLFW_KEY_ENTER] = GL_TRUE;
			}
			if (this->Keys[GLFW_KEY_W] && !this->KeysProcessed[GLFW_KEY_W]) {
				this->Level = (this->Level + 1) % 4;
				this->KeysProcessed[GLFW_KEY_W] = GL_TRUE;
			}
			if (this->Keys[GLFW_KEY_S] && !this->KeysProcessed[GLFW_KEY_S]) {
				if (this->Level > 0) {
					--this->Level;
				}
				else {
					this->Level = 3;
				}
				this->KeysProcessed[GLFW_KEY_S] = GL_TRUE;
			}
		}
	}
	if (this->States == GAME_WIN) {
		if (this->Keys[GLFW_KEY_ENTER]) {
			this->KeysProcessed[GLFW_KEY_ENTER] = GL_TRUE;
			Effects->Chaos = GL_FALSE;
			this->States = GAME_MENU;
		}
	}
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
			Ball->Stuck = GL_FALSE;
		}
	}
}

void Game::Render() {
	if (this->States == GAME_ACTIVE || this->States == GAME_MENU || this->States == GAME_WIN) {
		Effects->BeginRender();
		Renderer->DrawSprite(ResourceManager::GetTexture("background") ,glm::vec2(0 ,0) ,glm::vec2(this->Width ,this->Height) ,0.0f);
		this->Levels[this->Level].Draw(*Renderer);
		Player->Draw(*Renderer);
		for (PowerUp & powerUp : this->PowerUps) {
			if (!powerUp.Destroyed) {
				powerUp.Draw(*Renderer);
			}
		}
		Particles->Draw();
		Ball->Draw(*Renderer);
		Effects->EndRender();
		Effects->Render(glfwGetTime());
		std::stringstream ss;
		ss << this->Lives;
		Text->RenderText("Lives:" + ss.str() ,5.0f ,5.0f ,1.0f);
	}
	if (this->States == GAME_MENU) {
		Text->RenderText("Press ENTEr to start" ,250.0f ,this->Height /2 ,1.0f);
		Text->RenderText("Press W or S to select level" ,245.0f ,this->Height /2 + 20.0f ,0.75f);
	}
	if (this->States == GAME_WIN) {
		Text->RenderText("You WON!!" ,320.f ,this->Height /2 -20.0f ,1.0f ,glm::vec3(0.0f ,1.0f ,0.0f));
		Text->RenderText("Press ENTER to retry or ESC to quit" ,130.0f ,this->Height /2 ,1.0f ,glm::vec3(1.0f ,1.0f ,1.0f));
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
	this->Lives = 3;
}

void Game::ResetPlayer() {
	Player->Size = PLAY_SIZE;
	Player->Position = glm::vec2(this->Width /2 - PLAY_SIZE.x ,this->Height - PLAY_SIZE.y);
	Ball->Reset(Player->Position + glm::vec2(PLAY_SIZE .x/2 - BALL_RADIUS ,-(BALL_RADIUS * 2)) , INITIAL_BALL_VELOCITY);
	Effects->Chaos = Effects->Confus = GL_FALSE;
	Ball->PassThrough = Ball->Sticky = GL_FALSE;
	Player->Color = glm::vec3(1.0f);
	Ball->Color = glm::vec3(1.0f);
	vector<PowerUp>::iterator it;
	for (it = PowerUps.begin(); it != PowerUps.end(); ) {
		it = PowerUps.erase(it);
	}
}


void ActivatePowerUp(PowerUp& powerUp) {
	if (powerUp.Type == "speed") {
		Ball->Velocity *= 1.2;
	}
	else if (powerUp.Type == "sticky") {
		Ball->Sticky = GL_TRUE;
		Player->Color = glm::vec3(1.0f, 0.5f, 1.0f);
	}
	else if (powerUp.Type == "pass-through") {
		Ball->PassThrough = GL_TRUE;
		Ball->Color = glm::vec3(1.0f, 0.5f, 0.5f);
	}
	else if (powerUp.Type == "pad-size-increase") {
		Player->Size.x += 100;
	}
	else if (powerUp.Type == "confuse") {
		if (!Effects->Chaos) {
			Effects->Confus = GL_TRUE;
		}
	}
	else if (powerUp.Type == "chaos") {
		if (!Effects->Confus) {
			Effects->Chaos = GL_TRUE;
		}
	}
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
					this->SpawnPowerUps(box);
					SoundEngine->play2D("irrklang/bleep.mp3" ,GL_FALSE);
				}
				else {
					ShakeTime = 0.05f;
					Effects->Shake = GL_TRUE;
					SoundEngine->play2D("irrklang/solid.wav", GL_FALSE);
				}
				Direction dir = std::get<1>(collision);
				glm::vec2 diff_vector = std::get<2>(collision);
				if (!(Ball->PassThrough && !box.IsSolid)) {
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
	}
	for (PowerUp& powerUp : this->PowerUps) {
		if (!powerUp.Destroyed) {
			if (powerUp.Position.y >= this->Height) {
				powerUp.Destroyed = GL_TRUE;
			}
			if (CheckCollision(*Player, powerUp)) {
				ActivatePowerUp(powerUp);
				powerUp.Destroyed = GL_TRUE;
				powerUp.Activated = GL_TRUE;
				SoundEngine->play2D("irrklang/powerup.wav" ,GL_FALSE);
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

		Ball->Stuck = Ball->Sticky;
		SoundEngine->play2D("irrklang/bleep.wav" ,GL_FALSE);
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
		glm::vec2(1.0f ,0.0f),
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

GLboolean ShouldSpawn(GLuint chance) {
	GLuint random = rand() % chance;
	return random == 0;
}

void Game::SpawnPowerUps(GameObject &block) {
	// 1/ 75
	if (ShouldSpawn(5)) {
		this->PowerUps.push_back(PowerUp("speed" ,glm::vec3(0.5f ,0.5f ,1.0f) ,0.0f ,block.Position ,ResourceManager::GetTexture("powerup_speed")));
	}
	if (ShouldSpawn(5)) {
		this->PowerUps.push_back(PowerUp("sticky" ,glm::vec3(1.0f ,0.5f ,1.0f) ,20.0f ,block.Position, ResourceManager::GetTexture("powerup_sticky")));
	}
	if (ShouldSpawn(5)) {
		this->PowerUps.push_back(PowerUp("pass-through" ,glm::vec3(0.5f ,1.0f ,0.5f) ,10.0f ,block.Position, ResourceManager::GetTexture("powerup_passthrough")));
	}
	if (ShouldSpawn(5)) {
		this->PowerUps.push_back(PowerUp("pad-size-increase" ,glm::vec3(1.0f ,0.6f ,0.4f) ,0.0f ,block.Position, ResourceManager::GetTexture("powerup_increase")));
	}
	// 1 / 15
	if (ShouldSpawn(5)) {
		this->PowerUps.push_back(PowerUp("confuse", glm::vec3(1.0f, 0.3f, 0.3f), 15.0f, block.Position ,ResourceManager::GetTexture("powerup_confuse")));
	}
	if (ShouldSpawn(5)) {
		this->PowerUps.push_back(PowerUp("chaos" ,glm::vec3(0.9f ,0.25f ,0.25f) ,15.0f ,block.Position, ResourceManager::GetTexture("powerup_chaos")));
	}
}

GLboolean IsOtherPowerUpActive(std::vector<PowerUp>& powerUps, std::string type) {
	for (const PowerUp& powerUp : powerUps) {
		if (powerUp.Activated) {
			if (powerUp.Type == type) {
				return GL_TRUE;
			}
		}
	}
	return GL_FALSE;
}

void Game::UpdatePowerUp(GLfloat dt) {
	for (PowerUp &powerUp : this->PowerUps) {
		powerUp.Position += powerUp.Velocity * dt;
		if (powerUp.Activated) {
			powerUp.Duration -= dt;
			if (powerUp.Duration <= 0.0f) {
				powerUp.Activated = GL_FALSE;
				
				if (powerUp.Type == "sticky") {
					if (!IsOtherPowerUpActive(this->PowerUps ,"sticky")) {
						Ball->Sticky = GL_FALSE;
						Player->Color = glm::vec3(1.0f);
					}
				}
				else if (powerUp.Type == "pass-through") {
					if (!IsOtherPowerUpActive(this->PowerUps ,"pass-through")) {
						Ball->PassThrough = GL_FALSE;
						Ball->Color = glm::vec3(1.0f);
					}
				}
				else if (powerUp.Type == "confuse") {
					if (!IsOtherPowerUpActive(this->PowerUps ,"confuse")) {
						Effects->Confus = GL_FALSE;
					}
				}
				else if (powerUp.Type == "chaos") {
					if (!IsOtherPowerUpActive(this->PowerUps ,"chaos")) {
						Effects->Chaos = GL_FALSE;
					}
				}
			}
		}
		this->PowerUps.erase(std::remove_if(this->PowerUps.begin() ,this->PowerUps.end(),
			[](const PowerUp& powerUp) {return powerUp.Destroyed && !powerUp.Activated; })
			,this->PowerUps.end());
	}
}