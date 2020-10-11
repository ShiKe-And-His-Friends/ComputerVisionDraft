#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

void framebuffer_size_callback (GLFWwindow* window ,int width ,int height);
void processInput (GLFWwindow *window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

using namespace std;

const char *vertexShaderSource = "#version 330 core\n"
	"layout (location = 0) in vec3 aPos;\n"
	"void main () \n"
	"{ \n"
	"	gl_Position = vec4(aPos.x ,aPos.y ,aPos.z ,1.0);\n"
	"}\n\0";
const char* fragmentShaderSource = "#version 330 core\n"
	"out vec4 FragColor;\n"
	"void main () \n"
	"{ \n"
	"	FragColor = vec4(1.0f ,0.5f ,0.2f ,1.0f);\n"
	"}\n\0";

int main () {

	// Initialize and configure.
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowHint(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT ,GL_TRUE);
#endif

	// Windows create.
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH ,SCR_HEIGHT ,"Triangle" ,NULL ,NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window\n" <<endl;
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);

	// Load GLAD pointer.
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cout << "ailed to initialize GLAD\n" <<endl;
		return -1;
	}

	// Build and compile shader program
	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader ,1 ,&vertexShaderSource ,NULL);
	glCompileShader(vertexShader);
	// Check for shader compile error.
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader ,GL_COMPILE_STATUS ,&success);
	if (!success) {
		glGetShaderInfoLog(vertexShader ,512 ,NULL ,infoLog);
		cout << "ERROR::SHADER::BERTEX::COMPILATION_FAILED\n" << infoLog << endl;
	}
	// Fragment shader.
	int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader ,1 ,&fragmentShaderSource ,NULL);
	glCompileShader(fragmentShader);
	// Check for shader compile error.
	glGetShaderiv(fragmentShader ,GL_COMPILE_STATUS ,&success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader ,512 ,NULL ,infoLog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog <<endl;
	}

	// Link program.
	int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram ,vertexShader);
	glAttachShader(shaderProgram ,fragmentShader);
	glLinkProgram(shaderProgram);
	// Check for link program error.
	glGetProgramiv(shaderProgram ,GL_LINK_STATUS ,&success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram ,512 ,NULL ,infoLog);
		cout << "ERROR::SHADER::PROGRAME::LINKING_FAILED\n" << infoLog <<endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// Set up vertex data and buffer(s) and configure attributes.
	float vertices[] = {
		-0.5f ,-0.5f ,0.0f ,  	// left
		0.5f ,-0.5f ,0.0f ,  	// right
		0.0f ,0.5f ,0.0f 	// top
	};

	unsigned int VBO ,VAO;
	glGenVertexArrays(1 ,&VAO);
	glGenBuffers(1 ,&VBO);
	// Bind the vertex array pbject first ,then bind and set vertex buffer(s) ,and then configure\n
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,3 * sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);

	// Note that this is allowed ,the call to glVertexAttribPointer registed VBP as the vertex attribute.
	glBindBuffer(GL_ARRAY_BUFFER ,0);

	/**
	 * You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO ,but it rarely happens.
	 * Modify other VAOs requires a call to glBindVertexArray anyways so we generally don\t unbind VAOs(or VBO) when isn't directly necessary.
	 * */
	glBindVertexArray(0);

	// uncomment this call to draw in wireframe polygones.
	//glPolygonMode(GL_FRONT_AND_BACK ,GL_LINK);
	
	// Render loop
	while (!glfwWindowShouldClose(window)) {
		// Input
		processInput(window);

		// Render
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Draw our first triangle
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES ,0 ,3);
		//glBindVertexArray(0);
		
		// Swap buffers and IO events.
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Optional : de-allocate all resource once they've outlived their purpose.
	glDeleteVertexArrays(1 ,&VAO);
	glDeleteBuffers(1 ,&VBO);
	glDeleteProgram(shaderProgram);

	glfwTerminate();
	return 0;
}

void processInput (GLFWwindow* window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
}

void framebuffer_size_callback (GLFWwindow* window ,int width ,int height) {
	glViewport(0 ,0 ,width ,height);
}
