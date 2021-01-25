#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "raw/stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "raw/shader_s_2.h"
#include "raw/camera_2.h"
#include "raw/model.h"
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);
void mouse_callback(GLFWwindow* window ,double xpos ,double ypos);
void processInput(GLFWwindow* window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f ,0.0f ,3.0f));
float lastX = SCR_WIDTH /2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

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
		std::cout << "GLFW Window init failure." << std::endl;
		glfwTerminate();
		return -1;
	} else {
		std::cout << "GLFW Window init success." << std::endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);
	glfwSetCursorPosCallback(window ,mouse_callback);
	
	glfwSetInputMode(window ,GLFW_CURSOR ,GLFW_CURSOR_DISABLED);
	
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Initialize GLAD failure." << std::endl;
		return -1;
	} else {
		std::cout << "Initialize GLAD success." << std::endl;
	}
	
	glEnable(GL_DEPTH_TEST);
	
	Shader shaderRed("./raw/8.advanced_glsl.vs" ,"./raw/8.red.fs" );
	Shader shaderGreen("./raw/8.advanced_glsl.vs" ,"./raw/8.green.fs" );
	Shader shaderBlue("./raw/8.advanced_glsl.vs" ,"./raw/8.blue.fs" );
	Shader shaderYellow("./raw/8.advanced_glsl.vs" ,"./raw/8.yellow.fs" );

	float cubeVertices[] = {
		// positions         
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
	
	// cube VAO
    unsigned int cubeVAO, cubeVBO;
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), &cubeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);

	// uniform configuration
	// first. We get the relevant block indices
	unsigned int uniformBlockIndexRed = glGetUniformBlockIndex(shaderRed.ID, "Matrices");
	unsigned int uniformBlockIndexGreen = glGetUniformBlockIndex(shaderGreen.ID, "Matrices");
	unsigned int uniformBlockIndexBlue = glGetUniformBlockIndex(shaderBlue.ID, "Matrices");
	unsigned int uniformBlockIndexYellow = glGetUniformBlockIndex(shaderYellow.ID, "Matrices");
	// then we link each shader's uniform block to this uniform binding point
	glUniformBlockBinding(shaderRed.ID, uniformBlockIndexRed, 0);
	glUniformBlockBinding(shaderGreen.ID, uniformBlockIndexGreen, 0);
	glUniformBlockBinding(shaderBlue.ID, uniformBlockIndexBlue, 0);
	glUniformBlockBinding(shaderYellow.ID, uniformBlockIndexYellow, 0);
	// Now actually create the buffer
	unsigned int uboMatrices;
	glGenBuffers(1, &uboMatrices);
	glBindBuffer(GL_UNIFORM_BUFFER, uboMatrices);
	glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), NULL, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	// define the range of the buffer that links to a uniform binding point
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, uboMatrices, 0, 2 * sizeof(glm::mat4));

	// store the projection matrix (we only do this once now) (note: we're not using zoom anymore by changing the FoV)
	glm::mat4 projection = glm::perspective(45.0f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
	glBindBuffer(GL_UNIFORM_BUFFER, uboMatrices);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(projection));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	
	while(!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		
		processInput(window);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 view = camera.GetViewMatrix();
		glBindBuffer(GL_UNIFORM_BUFFER ,uboMatrices);
		glBufferSubData(GL_UNIFORM_BUFFER ,sizeof(glm::mat4) ,sizeof(glm::mat4) ,glm::value_ptr(view));
		glBindBuffer(GL_UNIFORM_BUFFER ,0);
		
		// draw 4 cubes
		glBindVertexArray(cubeVAO);
		shaderRed.use();
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model ,glm::vec3(-0.75 ,0.75 ,0.0f));
		shaderRed.setMat4("model" ,model);
		glDrawArrays(GL_TRIANGLES ,0 ,36);
		// GREEN
		shaderGreen.use();
		model = glm::mat4(1.0f);
		model = glm::translate(model ,glm::vec3(0.75 ,0.75 ,0.0f));
		shaderGreen.setModel("model" ,model);
		glDrawArrays(GL_TRIANGLES ,0 ,36);
		//YELLOW
		shaderYellow.use();
		model = glm::mat4(1.0f);
		model = glm::translate(model ,glm::vec3(-0.75f ,-0.75f ,0.0f));
		shaderYellow.setModel("model" ,model);
		glDrawArrays(GL_TRIANGLES ,0 ,36);
		// BLUE
		shaderBlue.use();
		model = glm::mat4(1.0f);
		model = glm::translate(model ,glm::vec3(-0.75f ,-0.75f ,0.0f));
		shaderBlue.setModel("model" ,model);
		glDrawArrays(GL_TRIANGLES ,0 ,36);
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	glDeleteVertexArrays(1 ,&cubeVAO);
	glDeleteBuffers(1 ,&cubeVBO);
	
	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
	if (glfwGetKey(window ,GLFW_KEY_W) == GLFW_PRESS) {
		camera.ProcessKeyBoard(FORWARD ,deltaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_S) == GLFW_PRESS) {
		camera.ProcessKeyBoard(BACKWARD ,deltaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_A) == GLFW_PRESS) {
		camera.ProcessKeyBoard(LEFT ,deltaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_D) == GLFW_PRESS) {
		camera.ProcessKeyBoard(RIGHT ,deltaTime);
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
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;
	camera.ProcessMouseMovement(xoffset ,yoffset);
}