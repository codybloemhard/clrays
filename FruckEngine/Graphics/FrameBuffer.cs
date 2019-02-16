using System;
using System.Collections.Generic;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace FruckEngine.Graphics {
    /// <summary>
    /// Framebuffer abstraction class
    /// </summary>
    public class FrameBuffer {
        private int Width, Height;
        public int Pointer, RBOPointer;
        public Dictionary<string, Texture> Attachments = new Dictionary<string, Texture>();
        private List<string> AttachmentOrder = new List<string>();

        /// <summary>
        /// Creates framebuffer and sets its base size
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        public FrameBuffer(int width, int height) {
            Width = width;
            Height = height;

            Pointer = GL.GenFramebuffer();
        }

        /// <summary>
        /// Adds an attachment(texture) to the framebuffer with given format. It is assumed that its a texture 2d
        /// </summary>
        /// <param name="name"></param>
        /// <param name="pixelType"></param>
        /// <param name="internalFormat"></param>
        /// <param name="format"></param>
        /// <param name="filter"></param>
        /// <param name="depth">Wether attachemnt is a depth attachment.</param>
        public void AddAttachment(string name, PixelType pixelType = PixelType.Float,
            PixelInternalFormat internalFormat = PixelInternalFormat.Rgba16f,
            PixelFormat format = PixelFormat.Rgba, TextureMinFilter filter = TextureMinFilter.Nearest,
            bool depth = false) {
            // Create a blank texure
            var texture = new Texture() {
                FilterMin = filter,
                FilterMag = (TextureMagFilter) filter,
                WrapS = TextureWrapMode.ClampToEdge,
                WrapT = TextureWrapMode.ClampToEdge,
                MipMap = false
            };
            texture.Load(Width, Height, internalFormat, format, TextureTarget.Texture2D, pixelType, (IntPtr) 0);

            // Attach the texture
            var attachment = depth
                ? FramebufferAttachment.DepthAttachment
                : FramebufferAttachment.ColorAttachment0 + AttachmentOrder.Count;
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, attachment, TextureTarget.Texture2D,
                texture.Pointer, 0);

            texture.UnBind();
            Attachments.Add(name, texture);
            AttachmentOrder.Add(name);
        }

        /// <summary>
        /// Add render buffer. By default used to add a renderbuffer for depth.
        /// Use this if you want to have depth in your frame buffer but dont want to render it to a texture.
        /// Dont forget to blit to copy depth if needed
        /// </summary>
        /// <param name="storage"></param>
        /// <param name="attachment"></param>
        public void AddRenderBuffer(RenderbufferStorage storage = RenderbufferStorage.DepthComponent,
            FramebufferAttachment attachment = FramebufferAttachment.DepthAttachment) {
            RBOPointer = GL.GenRenderbuffer();
            GL.BindRenderbuffer(RenderbufferTarget.Renderbuffer, RBOPointer);
            GL.RenderbufferStorage(RenderbufferTarget.Renderbuffer, storage, Width, Height);
            GL.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, attachment, RenderbufferTarget.Renderbuffer,
                RBOPointer);
        }

        /// <summary>
        /// Gets attachment texture
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Texture GetAttachment(string name) {
            return Attachments[name];
        }

        /// <summary>
        /// Name is bit misleading.  It basically tels the buffer which color buffers to use.
        /// Use this if you have more than one attachment after you attached all the textures.
        /// </summary>
        public void DrawBuffers() {
            var colorAttachments = new DrawBuffersEnum[AttachmentOrder.Count];
            int colorCounter = 0;
            for (int i = 0; i < AttachmentOrder.Count; i++) {
                if (AttachmentOrder[i] != "depth") {
                    colorAttachments[i] = DrawBuffersEnum.ColorAttachment0 + colorCounter++;
                }
            }

            GL.DrawBuffers(AttachmentOrder.Count, colorAttachments);
        }

        /// <summary>
        /// Bind the framebuffer. You also have the option to clear and change the viewport.
        /// <b>Warning:</b> clear also clears the depth buffer which might not always be beneficial when doing multiple passes
        /// Dont clear when you are still building the framebuffer or you will get a gl error that framebuffer is not yet complete
        /// </summary>
        /// <param name="clear"></param>
        /// <param name="setViewport"></param>
        public void Bind(bool clear = true, bool setViewport = false) {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, Pointer);
            if (clear) GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            if (setViewport) GL.Viewport(0, 0, Width, Height);
        }

        /// <summary>
        /// Unbind the frame buffer
        /// </summary>
        public void UnBind() {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        }

        /// <summary>
        /// Resize framebuffer and all its attachments
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        public void Resize(int width, int height) {
            Width = width;
            Height = height;
            foreach (var attachment in Attachments) attachment.Value.Resize(Width, Height);
        }

        /// <summary>
        /// Assert if framebuffer is complete
        /// </summary>
        /// <exception cref="Exception"></exception>
        public void AssertStatus() {
            if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete) {
                throw new Exception("Framebuffer not complete!");
            }
        }

        /// <summary>
        /// Detach an attachment. Dont forget to reattach something again
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Texture Detach(string name) {
            var ret = Attachments[name];
            Attachments.Remove(name);
            AttachmentOrder.Remove(name);
            return ret;
        }

        /// <summary>
        /// Shorthand or actually just easy to read. Render shit to a plane. Make sure to have appropriate shaders
        /// </summary>
        public void RenderToPlane() {
            Projection.ProjectPlane();
        }
    }
}