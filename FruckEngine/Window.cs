﻿using System;
using System.ComponentModel;
using System.Drawing;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace FruckEngine {
    /// <summary>
    /// Widnow holds the game and is responsible for passing the window state to the ghame
    /// </summary>
    public class Window : GameWindow {
        protected Game Game;
        protected double LastMouseX, LastMouseY, LastMouseScroll;
        protected bool FirstMouse = true;
        protected bool LockMouse = true;
        protected bool MouseDown = false;

        public Window(int width, int height, string title, Game game)
            : base(width, height, GraphicsMode.Default, title) {
            Game = game;
            //CursorVisible = false;
        }

        /// <summary>
        /// Load and initialize the game
        /// </summary>
        /// <param name="e"></param>
        protected override void OnLoad(EventArgs e) {
            UpdateViewport();
            Game.Init();
            Game.Clear();
            GL.Hint(HintTarget.PerspectiveCorrectionHint, HintMode.Nicest);
            // Enable depth test and seamless cubemap
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.TextureCubeMapSeamless);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="e"></param>
        protected override void OnUnload(EventArgs e) {
            Game.Destroy();
            Environment.Exit(0); // bypass wait for key on CTRL-F5
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="e"></param>
        protected override void OnResize(EventArgs e) {
            UpdateViewport();
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            base.OnClosing(e);
            Game.Destroy();
            Environment.Exit(0);
        }

        /// <summary>
        /// Update frame calls also all the input functions
        /// </summary>
        /// <param name="e"></param>
        protected override void OnUpdateFrame(FrameEventArgs e) {
            UpdateMouse();
            if (!Focused || !LockMouse) return;
            var state = Keyboard.GetState();
            Game.OnKeyboardUpdate(state);
            if (state[Key.Escape]) Exit();

            Game.Update(e.Time);
        }

        /// <summary>
        /// Track mouse state and send the changes to the game
        /// </summary>
        private void UpdateMouse() {
            var state = Mouse.GetCursorState();
            
            if (!FirstMouse) {
                if (state.X != LastMouseX || state.Y != LastMouseY) {
                    Game.OnMouseMove(state.X - LastMouseX, state.Y - LastMouseY);
                    Game.MousePosition = new Vector2(state.X - Bounds.Left, state.Y - Bounds.Top);
                }

                if (state.ScrollWheelValue != LastMouseScroll) {
                    Game.OnMouseScroll(state.ScrollWheelValue - LastMouseScroll);
                }
            }

            if (MouseDown != (state.LeftButton == ButtonState.Pressed)) {
                MouseDown = state.LeftButton == ButtonState.Pressed;
                Game.OnMouseButton(MouseDown);
            }

            LastMouseX = state.X;
            LastMouseY = state.Y;
            LastMouseScroll = state.WheelPrecise;


            FirstMouse = false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="e"></param>
        protected override void OnRenderFrame(FrameEventArgs e) {
            Game.Clear();
            Game.Render(e.Time);
            SwapBuffers();
        }

        /// <summary>
        /// Update viewport and flip the Y which seems to be flipped
        /// </summary>
        private void UpdateViewport() {
            if (Game.Width == Width && Game.Height == Height) return;
            GL.Viewport(0, 0, Width, Height);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadIdentity();
            GL.Ortho(-1.0, 1.0, -1.0, 1.0, 0.0, 4.0);
            Game.Resize(Width, Height);
        }
    }
}