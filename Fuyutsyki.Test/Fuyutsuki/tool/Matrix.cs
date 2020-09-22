using System;
using System.Threading.Tasks;

namespace Fuyutsuki
{
    [Serializable]
    public class Matrix
    {
        public int Row;
        public int Column;
        public double[,] Data;
        public Matrix()
        { }
        public Matrix(int row,int column)
        {
            Data = new double[row,column];
            Row = Data.GetLength(0);
            Column = Data.GetLength(1);

        }
        public static Matrix ZerosLike(int row, int column)
        {
            var result = new Matrix(row,column);
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                    result.Data[i, j] = 1;
            }
            return result;

        }
        public static Matrix ZerosLike(Matrix m)
        {
            var result = MatClone(m);
            for (int i = 0; i < result.Row; i++)
            {
                for (int j = 0; j < result.Column; j++)
                    result.Data[i, j] = 1;
            }
            return result;

        }
        public Matrix(double[,] data)
        {
            Data = data;
            Row = data.GetLength(0);
            Column = data.GetLength(1);

        }

        public static Matrix MatExp(Matrix a,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = Math.Exp(a.Data[i, j]);
                    }
                }

            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = Math.Exp(a.Data[i, j]);
                }
            });
            return result;
        }

        public Matrix(double data)
        {
            Data = new double[1, 1] { { data } };

            Row = 1;
            Column = 1;

        }
        

        /*行列の積算*/
        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.Column != b.Column)
            {
                if (a.Column == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatMul(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatMul(a, n);
                }
            }
            else if (a.Row != b.Row)
            {
                if (a.Row == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatMul(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatMul(a, n);
                }
            }
            else
                return MatMul(a, b);
        }
        public static Matrix operator *(Matrix a, double b)
        {
            return MatMul(a, b);
        }
        public static Matrix operator *(double b, Matrix a)
        {
            return MatMul(a, b);
        }
        public static Matrix MatDot(Matrix a, Matrix b,bool parallel=false)
        {
            int n = a.Column;
            var result = CreateMatrix(a.Row, b.Column);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < b.Column; j++)
                    {
                        double sum = 0;
                        for (int k = 0; k < n; k++)
                        {
                            var y= a.Data[i, k] * b.Data[k, j];
                            sum += y;
                        }
                        result.Data[i, j] = sum;

                    }
                }
            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < b.Column; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a.Data[i, k] * b.Data[k, j];
                    }
                    result.Data[i, j] = sum;

                }
            });
            return result;
        }
        public static Matrix MatMul(Matrix a, Matrix b,bool parallel=false)
        {
            int n = a.Column;
            var result = CreateMatrix(a.Row, a.Column);
            if (!parallel)
                for(int i = 0;i < a.Row;i++)
                {
                     for (int j = 0; j < a.Column; j++)
                {


                    result.Data[i, j] = a.Data[i, j] * b.Data[i, j];


                }
                }
            else
                 Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {


                    result.Data[i, j] = a.Data[i, j] * b.Data[i, j];


                }
            });
            return result;
        }
        public static Matrix MatMul(Matrix a, double b,bool parallel=false)
        {
            int n = a.Column;
            var result = CreateMatrix(a.Row, a.Column);
            if (!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = a.Data[i, j] * b;

                    }
                }
            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = a.Data[i, j] * b;

                }
            });
            return result;
        }
        public static Matrix CreateMatrix(int row, int column)
        {
            Matrix result = new Matrix(new double[row, column]);
            return result;


        }
        public static Matrix CreateMatrix(int column)
        {
            Matrix result = new Matrix(new double[1, column]);
            return result;


        }

        /*行列の除算*/
        public static Matrix operator /(Matrix a, Matrix b)
        {
            if (a.Column != b.Column)
            {
                if (a.Column == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatDiv(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatDiv(a, n);
                }
            }
            else if (a.Row != b.Row)
            {
                if (a.Row == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatDiv(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatDiv(a, n);
                }
            }
            else
                return MatDiv(a, b);
        }
        public static Matrix operator /(Matrix a, double b)
        {
            return MatDiv(a, b);
        }
        public static Matrix operator /(double a, Matrix b)
        {
            return MatDiv(a, b);
        }
        public static Matrix MatDiv(Matrix a, Matrix b,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        if (b.Data[i, j] != 0)
                            result.Data[i, j] = a.Data[i, j] / b.Data[i, j];
                        else
                            result.Data[i, j] = 0;
                    }
                }
            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    if (b.Data[i, j] != 0)
                        result.Data[i, j] = a.Data[i, j] / b.Data[i, j];
                    else
                        result.Data[i, j] = 0;
                }
            });
            return result;
        }
        public static Matrix MatDiv(Matrix a, double b,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {

                    result.Data[i, j] = a.Data[i, j] / b;

                }
            }

            else
            Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {

                    result.Data[i, j] = a.Data[i, j] / b;

                }
            });
            return result;
        }
        public static Matrix MatDiv( double b,Matrix a,bool parallel=false)
        {
            var result = MatClone(a);
            if (!parallel)
                for (int i = 0; i < a.Row; i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {

                        result.Data[i, j] = b / a.Data[i, j];

                    }
                }
            else
                Parallel.For(0, a.Row, i =>
           {
               for (int j = 0; j < a.Column; j++)
               {

                   result.Data[i, j] = b / a.Data[i, j];

               }
           });
            return result;
        }
        /*行列の乗算*/
        public static Matrix operator ^(Matrix a, double c)
        {
            return MatPow(a, c);
        }
        public static Matrix MatPow(Matrix a, double c,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = Math.Pow(a.Data[i, j], c);
                    }
                }
            else
                Parallel.For(0, a.Row, i =>
             {
                 for (int j = 0; j < a.Column; j++)
                 {
                     result.Data[i, j] = Math.Pow(a.Data[i, j], c);
                 }
             });
            return result;
        }
       
        /*行列の加算*/
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.Column != b.Column)
            {
                if (a.Column == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatAdd(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatAdd(a, n);
                }
            }
            else if (a.Row != b.Row)
            {
                if (a.Row == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatAdd(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatAdd(a, n);
                }
            }
            else
                return MatAdd(a, b);
        }
        public static Matrix operator +(Matrix a, double b)
        {
            return MatAdd(a, b);
        }
        public static Matrix operator +(double a, Matrix b)
        {
            return MatAdd(b, a);
        }
        public static Matrix MatAdd(Matrix a, Matrix b,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = a.Data[i, j] + b.Data[i, j];
                    }
                }

            else
                Parallel.For(0,a.Row,i=>
            {
                for (int j = 0; j<a.Column; j++)
                {
                    result.Data[i, j] = a.Data[i, j] + b.Data[i, j];
                }
            });
            return result;
        }
        public static Matrix MatAdd(Matrix a,double b,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = a.Data[i, j] + b;
                    }
                }
            else
                Parallel.For(0, a.Row, i =>
             {
                 for (int j = 0; j < a.Column; j++)
                 {
                     result.Data[i, j] = a.Data[i, j] + b;
                 }
             });
            return result;
        }



        /// <summary>
        /// aからbを引いたmatrixを作成する関数
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.Column != b.Column)
            {
                if (a.Column == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatSub(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatSub(a, n);
                }
            }
            else if (a.Row != b.Row)
            {
                if (a.Row == 1)
                {
                    var n = BloadCast(a, b.Row, b.Column);
                    return MatSub(n, b);
                }
                else
                {
                    var n = BloadCast(b, a.Row, a.Column);
                    return MatSub(a, n);
                }
            }
            else
                return MatSub(a, b);
        }
        public static Matrix operator -(Matrix a, double b)
        {
            var result = MatClone(a);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = a.Data[i, j] - b;
                }
            }
            return result;
        }
        public static Matrix operator -(double a, Matrix b)
        {
            var result = MatClone(b);
            for (int i = 0; i < b.Row; i++)
            {
                for (int j = 0; j < b.Column; j++)
                {
                    result.Data[i, j] = a-b.Data[i, j];
                }
            }
            return result;
        }
        public static Matrix operator -(Matrix a)
        {
            var result = MatClone(a);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                    result.Data[i, j] = a.Data[i, j]*-1;
            }
            return result;
        }
        public static Matrix MatSub(Matrix a, Matrix b,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
                    }
                }
            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
                }
            });
            return result;
        }
        public static Matrix MatNeg(Matrix a, bool parallel = false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = a.Data[i, j] * -1;
                    }
                }
            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = a.Data[i, j] * -1;
                }
            });
            return result;
        }

        public static Matrix MatSin(Matrix a,bool parallel=false)
        {
            var result = MatClone(a);
            if (!parallel)
                for(int i=0;i<a.Row;i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = Math.Sin(a.Data[i, j]);
                }
            }
            else
                Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = Math.Sin(a.Data[i, j]);
                }
            });
            return result;

        }
        public static Matrix MatCos(Matrix a,bool parallel=false)
        {
            var result = MatClone(a);
            if(!parallel)
                for(int i=0;i<a.Row;i++)
                {
                    for (int j = 0; j < a.Column; j++)
                    {
                        result.Data[i, j] = Math.Cos(a.Data[i, j]);
                    }
                }
            else
            Parallel.For(0, a.Row, i =>
            {
                for (int j = 0; j < a.Column; j++)
                {
                    result.Data[i, j] = Math.Cos(a.Data[i, j]);
                }
            });
            return result;

        }


        /*新しい行列を生成する関数*/
        /// <summary>
        /// 同じ形の行列を作成する
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Matrix MatClone(Matrix a)
        {
            Matrix result = new Matrix();
            result = CreateMatrix(a.Row, a.Column);
            return result;
        }
        /// <summary>
        /// 要素がすべて１の同じ形の行列を作る
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Matrix OnesLike(Matrix a)
        {
            Matrix result = new Matrix();
            result = CreateMatrix(a.Row, a.Column);
            for (int i = 0; i < result.Row; i++)
            {
                for (int j = 0; j < result.Column; j++)
                {
                    result.Data[i, j] = 1.0;
                }
            }
            return result;
        }

        public static Matrix RandN(int row, int column)
        {
            var result = CreateMatrix(row, column);
            var rand = new Random();
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    result.Data[i, j] = rand.NextDouble();
                }
            }
            return result;
        }
        public static Matrix RandN(int row, int column, Random rand)
        {
            var result = CreateMatrix(row, column);

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    result.Data[i, j] = rand.NextDouble();
                }
            }
            return result;
        }
        /// <summary>
        /// min以上max以下の乱数の入った行列を作る
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Matrix CreateNoise(double min,double max,Matrix a)
        {
            Random rand = new Random();
            var result = MatClone(a);
            for(int i=0;i<result.Row;i++)
            {
                for (int j = 0; j < result.Column; j++)
                    result.Data[i, j] =min+rand.NextDouble()*(max-min);
            }
            return result;

        }
        public double[,] ToDouble()
        {
            return Data;
        }
        public override string ToString()
        {
            string result = "";
            for (int i = 0; i < Row; i++)
            {
                result += "[ ";
                for (int j = 0; j < Column; j++)
                {
                    result += Data[i, j] + " ";
                }
                result += "]\r\n";
            }
            return result;
        }

        //形を変える関数

        /// <summary>
        /// xをrow*columnの行列に変換する
        /// </summary>
        /// <param name="x"></param>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public static Matrix Reshape(Matrix x, int row, int column)
        {
            var result = CreateMatrix(row, column);
            int num = 0;
            foreach (var data in x.Data)
            {
                result.Data[num / column, num % column] = data;
                num++;
            }
            return result;
        }
        /// <summary>
        /// 縦横を取り替えた行列を作成する
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Matrix MatTrans(Matrix x)
        {
            var result = CreateMatrix(x.Column, x.Row);
            for (int i = 0; i < x.Row; i++)
            {
                for (int j = 0; j < x.Column; j++)
                {
                    result.Data[j, i] = x.Data[i,j];
                }
            }
            return result;
        }
        
        /// <summary>
        /// 行もしくは列方向に合算する関数
        /// </summary>
        /// <param name="x">行列</param>
        /// <param name="axis">0:１列ごとに合算 1:１行毎に合算</param>
        /// <returns></returns>
        public static Matrix Sum(Matrix x,int? axis=null)
        {
            
            if (axis == null)
            {
                var result = CreateMatrix(1, 1);
                foreach (var data in x.Data)
                {
                    result.Data[0, 0] += data;
                }
                return result;
            }
            else if (axis == 0)
            {
                var result = CreateMatrix(1, x.Column);
                for (int i = 0; i < x.Row; i++)
                {
                    for (int j = 0; j < x.Column; j++)
                    {
                        result.Data[0, j] += x.Data[i, j];
                    }
                }
                return result;
            }

            else
            {
                var result = CreateMatrix(x.Row, 1);
                for (int i = 0; i < x.Row; i++)
                {
                    for (int j = 0; j < x.Column; j++)
                    {
                        result.Data[i, 0] += x.Data[i, j];
                    }
                }
                return result;
            }

           
        }
        /// <summary>
        /// 行列を広げる関数
        /// </summary>
        /// <param name="x"></param>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public static Matrix BloadCast(Matrix x, int row, int column)
        {
            var result = CreateMatrix(row, column);
            if (x.Row == row)
            {
                for (int i = 0; i < row; i++)
                {
                    for (int j = 0; j < column; j++)
                    {
                        result.Data[i, j] = x.Data[i, 0];
                    }
                }
            }
            else if (x.Column == column)
            {
                for (int i = 0; i < row; i++)
                {
                    for (int j = 0; j < column; j++)
                    {
                        result.Data[i, j] = x.Data[0, j];
                    }
                }
            }
            else
            {
                for (int i = 0; i < row; i++)
                {
                    for (int j = 0; j < column; j++)
                    {
                        result.Data[i, j] = x.Data[0, 0];
                    }
                }
            }
            return result;

        }
        public static Matrix SumTo(Matrix x, int row, int column)
        {
            var result = CreateMatrix(row, column);
            if (row == 1 && column == 1)
            {
                for (int i = 0; i < x.Row; i++)
                {
                    for (int j = 0; j < x.Column; j++)
                    {
                        result.Data[0, 0] += x.Data[i, j];
                    }
                }
            }
            else if (column==1)
            {
                for (int i = 0; i < x.Row; i++)
                {
                    for (int j = 0; j < x.Column; j++)
                    {
                        result.Data[i, 0] += x.Data[i, j];
                    }
                }
            }
            else if (row==1)
            {
                for (int i = 0; i < x.Row; i++)
                {
                    for (int j = 0; j < x.Column; j++)
                    {
                        result.Data[0, j] += x.Data[i, j];
                    }
                }
            }
            
            else
                result.Data = null;
            return result;

        }

        public static Matrix Sqrt(Matrix m,bool parallel=false)
        {
            var result = MatClone(m);
            if (!parallel)
                for (int i=0;i<m.Row;i++)
                {
                    for (int j = 0; j < m.Column; j++)
                        result.Data[i, j] = Math.Sqrt(m.Data[i, j]);
                }

            else
                Parallel.For(0, m.Row, i =>
            {
                for (int j = 0; j < m.Column; j++)
                    result.Data[i, j] = Math.Sqrt(m.Data[i, j]);
            });
            return result;
        }
        public static Matrix MatTanh(Matrix m, bool parallel=false)
        {
            var result = MatClone(m);
            if(!parallel)
            for(int i=0;i<m.Row;i++)
            {
                for (int j = 0; j < m.Column; j++)
                    result.Data[i, j] = Math.Tanh(m.Data[i, j]);
            }
            else
            Parallel.For(0, m.Row, i =>
            {
                for (int j = 0; j < m.Column; j++)
                    result.Data[i, j] = Math.Tanh(m.Data[i, j]);
            });
            return result;
        }

        public static Matrix MatLog(Matrix m,bool parallel=false)
        {
            var result = MatClone(m);
            if(!parallel)
            for(int i=0;i<m.Row;i++)
            {
                for (int j = 0; j < m.Column; j++)
                    result.Data[i, j] = Math.Log(m.Data[i, j]);
            }
            else
            Parallel.For(0, m.Row, i =>
            {
                for (int j = 0; j < m.Column; j++)
                    result.Data[i, j] = Math.Log(m.Data[i, j]);
            });
            return result;
        }
        /// <summary>
        /// n以下の要素をnに変える関数
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static Matrix Maximum(Matrix m,double n)
        {
            var result = MatClone(m);
            for (int i = 0; i < m.Row; i++)
            {
                for (int j = 0; j < m.Column; j++)
                {
                    if (n > m.Data[i, j])
                        result.Data[i, j] = n;
                    else
                        result.Data[i, j] = m.Data[i, j];
                }
                    
            }
            return result;
        }
        /// <summary>
        /// n以上の要素をnに変える
        /// </summary>
        /// <param name="m"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static Matrix Minimum(Matrix m,double n)
        {
            var result = MatClone(m);
            for (int i = 0; i < m.Row; i++)
            {
                for (int j = 0; j < m.Column; j++)
                {
                    if (m.Data[i, j] < n)
                        result.Data[i, j] = m.Data[i, j];
                    else
                        result.Data[i, j] = n;
                }
            }
            return result;
        }
        /// <summary>
        /// 行列の最大値を取得する関数
        /// </summary>
        /// <param name="x"></param>
        /// <param name="axis">null:すべて 0:縦方向 other:横方向</param>
        /// <returns></returns>
        public static Matrix Max(Matrix x, int? axis = null)
        {
            if (axis == null)
            {

                var max = x.Data[0, 0];
                foreach (var n in x.Data)
                {
                    if (max < n)
                        max = n;
                }
                return new Matrix(max);
            }
            else if(axis==0)
                {
                    var result = CreateMatrix(1, x.Column);
                    for (int i = 0; i < x.Column; i++)
                    {
                    var max = x.Data[0, i];
                        for (int j = 0; j < x.Row; j++)
                        {
                        if (max < x.Data[j, i])
                            max = x.Data[j, i];
                        }
                    result.Data[0, i] = max;
                    }
                return result;
                }
            else
            {
                var result = CreateMatrix(x.Row, 1);
                for (int i = 0; i < x.Row; i++)
                {
                    var max = x.Data[i, 0];
                    for (int j = 0; j < x.Column; j++)
                    {
                        if (max < x.Data[i, j])
                            max = x.Data[i, j];
                    }
                    result.Data[i, 0] = max;
                }
                return result;
            }
        }
        /// <summary>
        /// 行列の最小値を取得する関数
        /// </summary>
        /// <param name="x"></param>
        /// <param name="axis">null:すべて 0:縦方向 other:横方向</param>
        /// <returns></returns>
        public static Matrix Min(Matrix x, int? axis = null)
        {
            if (axis == null)
            {

                var max = x.Data[0, 0];
                foreach (var n in x.Data)
                {
                    if (max > n)
                        max = n;
                }
                return new Matrix(max);
            }
            else if (axis == 0)
            {
                var result = CreateMatrix(1, x.Column);
                for (int i = 0; i < x.Column; i++)
                {
                    var max = x.Data[0, i];
                    for (int j = 0; j < x.Row; j++)
                    {
                        if (max > x.Data[j, i])
                            max = x.Data[j, i];
                    }
                    result.Data[0, i] = max;
                }
                return result;
            }
            else
            {
                var result = CreateMatrix(x.Row, 1);
                for (int i = 0; i < x.Row; i++)
                {
                    var max = x.Data[i, 0];
                    for (int j = 0; j < x.Column; j++)
                    {
                        if (max > x.Data[i, j])
                            max = x.Data[i, j];
                    }
                    result.Data[i, 0] = max;
                }
                return result;
            }
        }

        public static Matrix LogSumExp(Matrix x, int? axis = 1)
        {
            var m = Max(x, axis);
            var y = x - m;
            y = MatExp(y);
            var s = Sum(y, axis);
            s = MatLog(s);
            return m + s;
        }
        /// <summary>
        /// vsに書かれたデータを抽出する
        /// </summary>
        /// <param name="vs">行数が書かれた１次元配列</param>
        /// <returns></returns>
        public Matrix Squeeze(int[] vs)
        {
            var result = CreateMatrix(vs.Length, this.Column);
            for (int i = 0; i < vs.Length; i++)
            {
                for (int j = 0; j < this.Column; j++)
                    result.Data[i, j] = this.Data[vs[i], j];
            }
            return result;
        }
        /*比較演算子*/
        public static Matrix operator >(Matrix a,double b)
        {
            var result = MatClone(a);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    if (a.Data[i, j] > b)
                        result.Data[i, j] = 1;
                    else
                        result.Data[i,j] = 0;
                }
            }
            return result;
        }
        public static Matrix operator >=(Matrix a, double b)
        {
            var result = MatClone(a);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    if (a.Data[i, j] >= b)
                        result.Data[i, j] = 1;
                    else
                        result.Data[i, j] = 0;
                }
            }
            return result;
        }
        public static Matrix operator <(Matrix a, double b)
        {
            var result = MatClone(a);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    if (a.Data[i, j] < b)
                        result.Data[i, j] = 1;
                    else
                        result.Data[i, j] = 0;
                }
            }
            return result;
        }
        public static Matrix operator <=(Matrix a, double b)
        {
            var result = MatClone(a);
            for (int i = 0; i < a.Row; i++)
            {
                for (int j = 0; j < a.Column; j++)
                {
                    if (a.Data[i, j] <= b)
                        result.Data[i, j] = 1;
                    else
                        result.Data[i, j] = 0;
                }
            }
            return result;
        }
        /// <summary>
        /// 行列をパディングする関数
        /// </summary>
        /// <param name="fR"></param>
        /// <param name="bR"></param>
        /// <param name="fC"></param>
        /// <param name="bC"></param>
        /// <returns></returns>
        public  Matrix MatPad( int fR, int fC)
        {
            var result = new Matrix(this.Row + fR + fR, this.Column + fC + fC);
            for (int i = 0; i < +this.Row; i++)
            {
                for (int j = 0; j < this.Column; j++)
                    result.Data[i + fR, j + fC] = this.Data[i, j];
            }
            return result;
            
        }

        public static int GetMaxIndex(Matrix x)
        {
            double maxnum = 0.0;
            int index = 0 ;
            for (int i = 0; i < x.Column; i++)
            {
                if (x.Data[0, i] > maxnum)
                {
                    index = i;
                    maxnum = x.Data[0, i];
                }
            }
            return index;

        }
    }
  

    
}
