#version 330 core
out vec4 outColor;

uniform sampler2D uTexture;

in VertexData {
    vec2 UV;
} i;

void main() {
    float color = texture(uTexture, i.UV).r * 1000;
    outColor = vec4(color, color, color, 1);
}
