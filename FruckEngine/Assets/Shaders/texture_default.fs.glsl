#version 330 core
out vec4 outColor;

uniform sampler2D uTexture;

in VertexData {
    vec2 UV;
} i;

void main() {
     outColor = vec4(texture(uTexture, i.UV).rgb, 1);
}
