#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "game.h"
#include "resource_manager.h"

void key_callback(GLFWwindow* window ,int key ,int scancode ,int action ,int mode);

const GLuint SCREEN_WIDTH = 800;
const GLuint SCREEN_HEIGHT = 600;

Game Breakout(SCREEN_WIDTH ,SCREEN_WIDTH);

int main(int argc ,char* argv[]) {
	glfwInit();
}

