#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

void framebuffer_size_callback(GLFWwindow *window ,int width ,int height);
void processInput(GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

int main()
{
	// Initialize envirement.
	glfwInit();
	glfwWindowInit(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowInit(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowInit(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);
	
	// Initialize windows.
	GLFWwindow *window = glfwCreateWindow(SCR_WIDTH ,SCR_HEIGHT ,"GLFW 3.3.2" ,NULL ,NULL);
	if (window == NULL) {
		std::count << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	// Notify current window bind current thread.
	glfwMakeContextCurrent(window);
	// Set Callback function of widnow that nofity function when size change.
	glfwSetFramwbufferSizeCallback(window ,framebuffer_size_callback);
	// Initialize GLAD mamage opengl pointer.
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	// Drawing loop.
	while (!glfwWindowShouldClose(window)) {
		// Input
		processInput(window);
		// Drawing command.
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GK_COLOR_BUFFER_BIT);
		// Check callback event ,exchange buffer
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Release/Delete allocated resource
	glfwTerminate();

	return EXIT_SUCCESS;
}

// Input manage that check user press 'ESC'
void processInput (GLFWwindow *window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
}

// Change view's size while windows's size through dragged
void framebuffer_size_callback (GLFWwindow *window ,int width ,int height) {
	//Tip: The width and height can larger than themself.Beacause human eys's retina.
	glViewport(0 ,0 ,width ,height);
}
