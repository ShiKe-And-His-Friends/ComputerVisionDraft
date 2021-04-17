#ifndef POWER_UP_H
#define POWER_UP_H

#include <string>
#include <GL/eglew.h>
#include <glm/glm.hpp>

#include "game_object.hpp"

const glm::vec2 SIZE_POWER_UP(60 ,20);
const glm::vec2 VELOCITY_POWER_UP(0.0f ,150.0f);

class PowerUp : public GameObject {

public:
	std::string Type;
	GLfloat Duration;
	GLboolean Activated;
	PowerUp(std::string type ,glm::vec3 color ,GLfloat duration ,glm::vec2 position ,Texture2D texture) 
		: GameObject(position , SIZE_POWER_UP,texture ,color , VELOCITY_POWER_UP) ,Type(type) ,Duration(duration) ,Activated(){
		this->Type = type;
		this->Color = color;
		this->Duration = duration;
		this->Position = position;
		this->Sprite = texture;
	}
};

#endif