#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);
void processInput(GLFWwindow* windows);

const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

int main () {

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowHint(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT ,GL_TRUE);
	cout << "Initialize Forward apple compat." << endl;
#endif

	GLFWwindow* window = glfwCreateWindow(800 ,600 ,"LearnOpenGl" ,NULL ,NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	} else {
		cout << "Success to create GLFW window" << endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cout << "Failed to initialize GLAD" << endl;
		return -1;
	} else {
		cout << "Success initialize GLAD" << endl;	
	}

	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
}

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height) {
	glViewport(0 ,0 ,width ,height);
}
