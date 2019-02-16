#version 330 core
out vec4 outColor;

in VertexData {
    vec2 UV;
} i;

void main() {
     outColor = vec4(i.UV.xy, 1, 1);
}
