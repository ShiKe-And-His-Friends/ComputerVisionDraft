#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "raw/stb_image.h"

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
unsigned int loadTexture(const char* path);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f ,0.0f ,3.0f));
float lastX = SCR_WIDTH /2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3 lightPos(1.2f ,1.0f ,2.0f);

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
	glfwSetScrollCallback(window ,scroll_callback);
	
	glfwSetInputMode(window ,GLFW_CURSOR ,GLFW_CURSOR_DISABLED);
	
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Initialize GLAD failure." << std::endl;
		return -1;
	} else {
		std::cout << "Initialize GLAD success." << std::endl;
	}
	glEnable(GL_DEPTH_TEST);
	
	Shader lightingShader("./raw/4.1.lighting_maps.vs" ,"./raw/4.1.lighting_maps_exerices_1.fs");
	Shader lightCubeShader("./raw/4.1.light_cube.vs" ,"./raw/4.1.light_cube.fs");
	
	float vertices[] = {
		// position 		// normals			// texture coords
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
	};
	
	unsigned int VBO ,cubeVAO;
	glGenVertexArrays(1 ,&cubeVAO);
	glGenBuffers(1 ,&VBO);
	
	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);
	
	glBindVertexArray(cubeVAO);
	
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,8*sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1 ,3 ,GL_FLOAT ,GL_FALSE ,8*sizeof(float) ,(void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2 ,2 ,GL_FLOAT ,GL_FALSE ,8*sizeof(float) ,(void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	
	unsigned int lightCubeVAO;
	glGenVertexArrays(1 ,&lightCubeVAO);
	glBindVertexArray(lightCubeVAO);
	
	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,8*sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	
	unsigned int diffuseMap = loadTexture("/home/shike/Documents/computerVisionDraft/202008/opengl_project/learnopengl_cn_github_io/drawable/container2.png");
	
	lightingShader.use();
	lightingShader.setInt("material.diffuse" ,0);
	
	while(!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		
		processInput(window);
		
		glClearColor(0.1f ,0.1f ,0.1f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		lightPos.x = 1.0f + sin(glfwGetTime()) * 2.0f;
		lightPos.y = sin(glfwGetTime() / 2.0f) * 1.0f;

		// be sure to active shader when setting uniforms/drawing objects
		lightingShader.use();
		lightingShader.setVec3("light.position" ,lightPos);
		lightingShader.setVec3("viewPos" ,camera.Position);
		
		// light properties
		lightingShader.setVec3("light.ambient" ,0.2f ,0.2f ,0.2f);
		lightingShader.setVec3("light.diffuse" ,0.5  ,0.5f ,0.5f);
		lightingShader.setVec3("light.specular" ,1.0f ,1.0f ,1.0f);
		
		// light properties
		lightingShader.setVec3("material.specular" ,0.5  ,0.5f ,0.5f);
		lightingShader.setFloat("material.shininess" ,64.0f);
		
		// view/projection transformations
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom) ,(float)SCR_WIDTH/(float)SCR_HEIGHT ,0.1f ,100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		lightingShader.setMat4("projection" ,projection);
		lightingShader.setMat4("view" ,view);
		
		glm::mat4 model = glm::mat4(1.0f);
		lightingShader.setMat4("model" ,model);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D ,diffuseMap);
		
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

void scroll_callback(GLFWwindow* window ,double xoffset ,double yoffset) {
	camera.ProcessMouseScroll(yoffset);
}

unsigned int loadTexture(char const* path) {
	unsigned int textureID;
	glGenTextures(1 ,&textureID);
	
	int width ,height ,nrComponents;
	unsigned char* data = stbi_load(path ,&width ,&height ,&nrComponents ,0);
	if (data) {
		GLenum format;
		if (nrComponents == 1) {
			format = GL_RED;
		} else if (nrComponents == 3) {
			format = GL_RGB;
		} else if (nrComponents == 4) {
			format = GL_RGBA;
		}
		
		glBindTexture(GL_TEXTURE_2D ,textureID);
		glTexImage2D(GL_TEXTURE_2D ,0 ,format ,width ,height ,0 ,format ,GL_UNSIGNED_BYTE ,data);
		glGenerateMipmap(GL_TEXTURE_2D);
		
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_S ,GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_T ,GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MIN_FILTER ,GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MAG_FILTER ,GL_LINEAR_MIPMAP_LINEAR);
	} else {
		std::cout << "Texture failed to load at path: " << path << std::endl;
		stbi_image_free(data);
	}
	return textureID;
}
