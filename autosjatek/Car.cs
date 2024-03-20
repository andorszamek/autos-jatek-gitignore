using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace autosjatek
{
    internal class Car
    {
       

        public int PosX { get; private set; }
        public int PosY { get; private set; }
        public int speed { get; private set; }
        private double acceleration { get; set; }
        public System.Windows.Shapes.Rectangle Rectangle { get; set; }
        public Car(int posX, int posY, int speed, double acceleration)
        {
            PosX = posX;
            PosY = posY;
            this.speed = speed;
            this.acceleration = acceleration;
        }
        public void Move(bool fasz)
        {
            if (fasz)
            {
                PosX += 10;
                 
            }
            else
            {
                PosX -= 10;
            }
        }
    }
}
