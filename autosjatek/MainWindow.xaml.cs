using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace autosjatek
{
   
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        [DllImport("kernel32.dll")]
        public static extern void AllocConsole();
        private Car jatekosauto ;
        private List<UIElement> CarDisplays= new List<UIElement>();
        private int Lanes = 4;
        public MainWindow()
        {
            InitializeComponent();
            AllocConsole();
            jatekosauto = new((int)(mycanvas.Width / 2), (int)(mycanvas.Height * 0.8), 100, 1);
            CreateRoad(Lanes);
            autoletrehozas(jatekosauto);

        }
        private void CreateRoad(int lanecount)
        {
            double lanewidth=mycanvas.Width/lanecount;
            for (int i = 0; i <= lanecount; i++)
            {
                Rectangle lane = new Rectangle();
                lane.Width = 10;
                lane.Height = mycanvas.Height;
                lane.Stroke = Brushes.White;
                lane.Fill = new SolidColorBrush(Color.FromArgb(255, 255, 255, 255));
                lane.StrokeThickness = 2;
                Canvas.SetLeft(lane, lanewidth*i);
                Canvas.SetTop(lane, 0);
                mycanvas.Children.Add(lane);
            }

        }
        private void autoletrehozas(Car car)
        {
            Rectangle rectfasz = new Rectangle();
            rectfasz.Width = 100;
            rectfasz.Height = 100;
            rectfasz.Stroke = Brushes.Blue;
            rectfasz.Fill = new SolidColorBrush(Color.FromArgb(255,0,0,255));
            rectfasz.StrokeThickness = 2;
            Canvas.SetLeft(rectfasz, car.PosX);
            Canvas.SetTop(rectfasz, car.PosY);
            jatekosauto.Rectangle = rectfasz;
            CarDisplays.Add(rectfasz);
            mycanvas.Children.Add(rectfasz);
        }
        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Right:
                    jatekosauto.Move(true);
                    Console.WriteLine(jatekosauto.PosX);
                    Canvas.SetLeft(CarDisplays[0], jatekosauto.PosX);
                    break;
                case Key.Left:
                    jatekosauto.Move(false);
                    Console.WriteLine(jatekosauto.PosX);
                    Canvas.SetLeft(CarDisplays[0], jatekosauto.PosX);
                    break;
                default: break;
            }
            
        }
        
        
    }
}
