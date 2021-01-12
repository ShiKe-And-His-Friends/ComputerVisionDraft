#include <glad/glad.h>
#include <GLFW/glfw3.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "raw/shader_s_2.h"
#include "raw/camera_2.h"
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);
void mouse_callback(GLFWwindow* window ,double xpos ,double ypos);
void scroll_callback(GLFWwindow* window ,double xoffset ,double yoffset);
void processInput(GLFWwindow* window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f ,0.0f ,3.0f));
float lastX = SCR_WIDTH /2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3 lighrPos(1.2f ,1.0f ,2.0f);

int main () {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowHint(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);
	
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT ,GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH ,SCR_HEIGHT ,"LearnOpenGL" ,NULL ,NULL);
	if (window == NULL) {
		std::out << "GLFW Window init failure." << std::endl;
		glfwTerminate();
		return -1;
	} else {
		std::out << "GLFW Window init success." << std::endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);
	glfwSetCursorPosCallback(window ,mouse_callback);
	glfwSetScrollCallback(window ,scroll_callback);
	
	glfwSetInoutMode(window ,GLFW_CURSOR ,GLFW_CURSOR_DISABLED);
	
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::out << "Initialize GLAD failure." << std::endl;
		return -1;
	} else {
		std::out << "Initialize GLAD success." << std::endl;
	}
	glEnable(GL_DEPTH_TEST);
	
	Shader lightingShader("./raw/1.colors.vs" ,"./raw/1.colors.fs");
	Shader lightCubeShader("./raw/1.light_cube.vs" ,"./raw/1.light_cube.fs");
	
	float vertices[] = {
		-0.5f, -0.5f, -0.5f, 
         0.5f, -0.5f, -0.5f,  
         0.5f,  0.5f, -0.5f,  
         0.5f,  0.5f, -0.5f,  
        -0.5f,  0.5f, -0.5f, 
        -0.5f, -0.5f, -0.5f, 

        -0.5f, -0.5f,  0.5f, 
         0.5f, -0.5f,  0.5f,  
         0.5f,  0.5f,  0.5f,  
         0.5f,  0.5f,  0.5f,  
        -0.5f,  0.5f,  0.5f, 
        -0.5f, -0.5f,  0.5f, 

        -0.5f,  0.5f,  0.5f, 
        -0.5f,  0.5f, -0.5f, 
        -0.5f, -0.5f, -0.5f, 
        -0.5f, -0.5f, -0.5f, 
        -0.5f, -0.5f,  0.5f, 
        -0.5f,  0.5f,  0.5f, 

         0.5f,  0.5f,  0.5f,  
         0.5f,  0.5f, -0.5f,  
         0.5f, -0.5f, -0.5f,  
         0.5f, -0.5f, -0.5f,  
         0.5f, -0.5f,  0.5f,  
         0.5f,  0.5f,  0.5f,  

        -0.5f, -0.5f, -0.5f, 
         0.5f, -0.5f, -0.5f,  
         0.5f, -0.5f,  0.5f,  
         0.5f, -0.5f,  0.5f,  
        -0.5f, -0.5f,  0.5f, 
        -0.5f, -0.5f, -0.5f, 

        -0.5f,  0.5f, -0.5f, 
         0.5f,  0.5f, -0.5f,  
         0.5f,  0.5f,  0.5f,  
         0.5f,  0.5f,  0.5f,  
        -0.5f,  0.5f,  0.5f, 
        -0.5f,  0.5f, -0.5f, 
	};
	
	unsigned int VBO ,cubeVAO;
	glGenVertexArrays(1 ,&cubeVAO);
	glGenBuffers(1 ,&VBO);
	
	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);
	
	glBindVertextArray(cubeVAO);
	
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,3*sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	
	unsigned int lightCubeVAO;
	glGenVertexArrays(1 ,&lightCubeVAO);
	glBindVertextArray(lightCubeVAO);
	
	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,3*sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	
	while(!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		
		processInput(window);
		
		glClearColor(0.1f ,0.1f ,0.1f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		lightingShader.use();
		lightingShader.setVec3("objectColor" ,1.0f ,0.5f ,0.31f);
		lightingShader.setVec3("lightColor" ,1.0f ,1.0f ,1.0f);
		
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom) ,(float)SCR_WIDTH / (float)SCR_HEIGHT ,0.1f ,100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		lightingShader.setMat4("projection" ,projection);
		lightingShader.setMat4("view" ,view);
		
		glm::mat4 model = glm::mat4(1.0f);
		lightingShader.setMat4("model" ,model);
		
		glBindVertexArray(cubeVAO);
		glDrawArrays(GL_TRIANGLES ,0 ,36);
		
		lightCubeShader.use();
		lightCubeShader.setMat4("projection" ,projection);
		lightCubeShader.setMat4("view" ,view);
		model = glm::mat4(1.0f);
		model = glm::translate(model ,lightPos);
		model = glm::scale(model ,glm::vec3(0.2f));
		lightCubeShader.setMat4("model" ,model);
		
		glBindVertexArray(lightCubeVAO);
		glDrawArrays(GL_TRIANGLES ,0 ,36);
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	glDeleteVertexArrays(1 ,&cubeVAO);
	glDeleteVertexArrays(1 ,&lightCubeVAO);
	glGenBuffers(1 ,&VBO);
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
	if (glfwGetKey(window ,GLFW_KEY_W) == GLFW_PRESS) {
		camera.ProcessKeyboard(FORWARD ,deltaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_S) == GLFW_PRESS) {
		camera.ProcessKeyboard(BACKWARD ,deltaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_A) == GLFW_PRESS) {
		camera.ProcessKeyboard(LEFT ,deltaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_D) == GLFW_PRESS) {
		camera.ProcessKeyboard(RIGHT ,deltaTime);
	}
}

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height) {
	glViewport(0 ,0 ,width ,height);
}

void mouse_callback(GLFWwindow* window ,double xpos ,double ypos) {
	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastX - ypos;
	lastX = xpos;
	lastY = ypos;
	camera.ProcessMouseMovement(xoffset ,yoffset);
}

void scroll_callback(GLFWwindow* window ,double xoffset ,double yoffset) {
	camera.ProcessMouseScroll(yoffset);
}