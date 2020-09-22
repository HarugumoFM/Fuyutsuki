using System;
using System.Collections.Generic;
using System.Linq;
using static Fuyutsuki.Matrix;

namespace Fuyutsuki
{
    [Serializable]
    public class Variable
    {
        public Matrix Weights;
       
        public Matrix Grads;
        public int Row { 
            get{ 
                if (_row == 0)
                { 
                    _row = Weights.Row;
                }
                return _row; 
            }
            set { this._row = value; }
        }
        private int _row;
        public int Column
        {
            get
            {
                if (_column == 0)
                {
                    _column = Weights.Column;
                }
                return _column;
            }
            set { this._column = value; }
        }
        private int _column;
        public List<Action> BackProp = new List<Action>();
        public Variable()
        { }
        /// <summary>
        /// Variableのコンストラクタ
        /// </summary>
        /// <param name="row">Variable's Row</param>
        /// <param name="column">Variable's Column</param>
        /// <param name="notZero">新しいVariableのWeightを０とするか</param>
        public Variable(int row,int column,bool notZero=false)
        {
            this.Row = row;
            this.Column = column;
            if (notZero)
                this.Weights = RandN(row, column);
            else
                this.Weights = CreateMatrix(row, column);
        }
        /// <summary>
        /// VariableのWeightsの内容を表示する
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return this.Weights.ToString();
        }
        /// <summary>
        /// 行列の形状を出力する関数
        /// </summary>
        /// <returns></returns>
        public (int, int) Shape()
        {
            return (this.Row, this.Column);
        }
       
        public Variable(Matrix a)
        {
            Weights = a;
            this.Row = a.Row;
            this.Column = a.Column;

        }
        public Variable(double a)
        {
            Weights = new Matrix(a);
            Row = 1;
            Column = 1;
        }
        public Variable(double[,] a)
        {
            Weights = new Matrix(a);
            Row = Weights.Row;
            Column = Weights.Column;
        }
        public double OutPut(int row, int column)
        {
            return this.Weights.Data[row, column];
        }
        public void BackWard()
        {
            for (int i = BackProp.Count - 1; i >= 0; i--)
            {
                this.BackProp[i]();

            }
            BackProp.Clear();
        }
        public void ClaerGrads()
        {
            this.Grads = null;
        }
        public static Variable Clone(Variable m)
        {
            var result = new Variable();
            result.Weights = MatClone(m.Weights);
            return result;
        }

        /*Variableの演算関数とオペレーター*/

        /// <summary>
        /// 二乗する関数
        /// </summary>
        /// <param name="a"></param>
        /// <returns>Variable</returns>
        public static Variable Square(Variable a)
        {
            var result = Clone(a);
            result.Weights = a.Weights * a.Weights;
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            Action Backward = () =>
            {
                if (result.Grads == null)
                {
                    result.Grads = OnesLike(result.Weights);
                }
                if (a.Grads == null)
                    a.Grads = 2.0 * a.Weights * result.Grads;
                else
                    a.Grads += 2.0 * a.Weights * result.Grads;

            };
            result.BackProp.Add(Backward);


            return result;
        }
        /// <summary>
        /// 各要素をeのa[,]乗するする関数
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Variable Exp(Variable a)
        {
            var result = Clone(a);
            result.Weights = MatExp(a.Weights);
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            Action Backward = () =>
            {
                if (result.Grads == null)
                {
                    result.Grads = OnesLike(result.Weights);
                }
                if (a.Grads == null)
                    a.Grads = MatExp(a.Weights) * result.Grads;
                else
                    a.Grads = a.Grads + (MatExp(a.Weights) * result.Grads);
            };
            result.BackProp.Add(Backward);
            return result;
        }

        public static Variable Sin(Variable a)
        {
            var result = Clone(a);
            result.Weights = MatSin(a.Weights);
            foreach (var item in a.BackProp)
            {
                if (!a.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
              {
                  if (result.Grads == null)
                      result.Grads = OnesLike(result.Weights);
                  if (a.Grads == null)
                      a.Grads = MatClone(a.Weights);
                  a.Grads += result.Grads * MatCos(a.Weights);
              };
            result.BackProp.Add(action);
            return result;

        }

        public static Variable Log(Variable a)
        {
            var result = Clone(a);
            result.Weights = MatLog(a.Weights);
            foreach (var item in a.BackProp)
            {
                if (!a.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                a.Grads += result.Grads / a.Weights;
            };
            result.BackProp.Add(action);
            return result;
        }

        //加算
        public static Variable operator +(Variable a, Variable b)
        {
            if (a.Weights.Column != b.Weights.Column)
            {
                if (a.Weights.Column == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Add(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Add(a, n);
                }
            }
            else if (a.Weights.Row != b.Weights.Row)
            {
                if (a.Weights.Row == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Add(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Add(a, n);
                }
            }
            else
                return Add(a, b);
        }
        public static Variable operator +(Variable a, double b)
        {

            return Add(a, b);
        }
        public static Variable operator +(double a, Variable b)
        {

            return Add(b, a);
        }

        public static Variable Add(Variable a, Variable b)
        {
            if(a.Row!=b.Row||a.Column!=b.Column)
                throw new NotImplementedException("Variable Sizes not match");
            var result = new Variable();
            result.Weights = a.Weights + b.Weights;
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            foreach (var action in b.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action backward = () =>
            {
                if (result.Grads == null)
                {
                    result.Grads = OnesLike(result.Weights);
                }
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                a.Grads = result.Grads + a.Grads;
                b.Grads = result.Grads + b.Grads;

            };
            result.BackProp.Add(backward);
            //BackProp.Add(backward);
            return result;
        }
        public static Variable Add(Variable a, double b)
        {
            var result = new Variable();
            result.Weights = a.Weights + b;
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            Action backward = () =>
            {
                if (result.Grads == null)
                {
                    result.Grads = OnesLike(result.Weights);
                }
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                a.Grads = result.Grads + a.Grads;

            };
            result.BackProp.Add(backward);
            //BackProp.Add(backward);
            return result;
        }

        //掛け算
        public static Variable operator *(Variable a, Variable b)
        {
            if (a.Weights.Column != b.Weights.Column)
            {
                if (a.Weights.Column == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Mul(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Mul(a, n);
                }
            }
            else if (a.Weights.Row != b.Weights.Row)
            {
                if (a.Weights.Row == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Mul(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Mul(a, n);
                }
            }
            else
                return Mul(a, b);
        }

        public static Variable operator *(Variable a, double b)
        {
            return Mul(a, b);
        }

        public static Variable operator *(double a, Variable b)
        {
            return Mul(b, a);
        }

        public static Variable Mul(Variable a, Variable b)
        {
            if (a.Row != b.Row || a.Column != b.Column)
                throw new NotImplementedException("(Mul)Variable Sizes not match");
            var result = new Variable();
            result.Weights = a.Weights * b.Weights;
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            foreach (var action in b.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action backward = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                a.Grads += result.Grads * b.Weights;
                b.Grads += result.Grads * a.Weights;

            };
            result.BackProp.Add(backward);
            return result;
        }


        public static Variable Mul(Variable a, double b)
        {
            var result = new Variable();
            result.Weights = a.Weights * b;
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            Action backward = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                a.Grads = (result.Grads * b) + a.Grads;

            };
            result.BackProp.Add(backward);
            return result;
        }
        public static Variable Mul(double c, Variable a)
        {
            return Mul(a, c);
        }
        //Dot算
        public static Variable Dot(Variable a, Variable b)
        {
            if ( a.Column != b.Row)
                throw new NotImplementedException("(Dot)Variable Sizes not match");
            var result = new Variable();
            result.Weights = MatDot(a.Weights, b.Weights);
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            foreach (var item in b.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                a.Grads += MatDot(result.Grads, Matrix.MatTrans(b.Weights));
                b.Grads += MatDot(Matrix.MatTrans(a.Weights), result.Grads);
            };
            result.BackProp.Add(action);
            return result;

        }

        //引き算

        public static Variable operator -(Variable a)
        {
            return Neg(a);
        }

        public static Variable Neg(Variable a)
        {
            var Result = Clone(a);
            Result.Weights = MatNeg(a.Weights);
            foreach (var func in a.BackProp)
            {
                if (!Result.BackProp.Contains(func))
                    Result.BackProp.Add(func);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (Result.Grads == null)
                    Result.Grads = OnesLike(Result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(Result.Grads);
                a.Grads -= Result.Grads;
            };
            return Result;
        }


        //割り算
        public static Variable operator /(Variable a, Variable b)
        {
            if (a.Weights.Column != b.Weights.Column)
            {
                if (a.Weights.Column == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Div(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Div(a, n);
                }
            }
            else if (a.Weights.Row != b.Weights.Row)
            {
                if (a.Weights.Row == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Div(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Div(a, n);
                }
            }
            else
                return Div(a, b);
        }
        public static Variable operator /(double a, Variable b)
        {

            return Div(a, b);
        }
        public static Variable operator /(Variable a, double b)
        {

            return Mul(a, 1 / b);
        }
        public static Variable Div(Variable a, Variable b)
        {
            if (a.Row != b.Row || a.Column != b.Column)
                throw new NotImplementedException("(Div)Variable Sizes not match");
            var result = Clone(a);
            result.Weights = a.Weights / b.Weights;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            foreach (var item in b.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = MatClone(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                a.Grads += result.Grads / b.Weights;
                b.Grads += result.Grads / a.Weights;

            };
            result.BackProp.Add(action);
            return result;

        }
        public static Variable Div(double a, Variable b)
        {
            var result = Clone(b);
            result.Weights = a / b.Weights;
            foreach (var item in b.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            b.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = MatClone(result.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                b.Grads += result.Grads / a;

            };
            result.BackProp.Add(action);
            return result;

        }


        //乗算
        public static Variable operator ^(Variable a, double c)
        {
            return Pow(a, c);
        }
        /// <summary>
        /// aのc乗をする関数
        /// </summary>
        /// <param name="a"></param>
        /// <param name="c">double</param>
        /// <returns></returns>
        ///
        public static Variable Pow(Variable a, double c)
        {
            var result = Clone(a);
            result.Weights = a.Weights ^ c;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);

                a.Grads += c * (a.Weights ^ (c - 1)) * result.Grads;


            };
            result.BackProp.Add(action);
            return result;

        }



        //除算
        public static Variable operator -(Variable a, Variable b)
        {
            if (a.Weights.Column != b.Weights.Column)
            {
                if (a.Weights.Column == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Sub(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Sub(a, n);
                }
            }
            else if (a.Weights.Row != b.Weights.Row)
            {
                if (a.Weights.Row == 1)
                {
                    var n = BloadCastTo(a, b.Weights.Row, b.Weights.Column);
                    return Sub(n, b);
                }
                else
                {
                    var n = BloadCastTo(b, a.Weights.Row, a.Weights.Column);
                    return Sub(a, n);
                }
            }
            else
                return Sub(a, b);
        }

        public static Variable operator -(Variable a, double c)
        {
            return Sub(a, c);
        }
        public static Variable operator -(double c, Variable a)
        {
            return Sub(c, a);
        }

        public static Variable Sub(Variable a, Variable b)
        {
            if (a.Row != b.Row || a.Column != b.Column)
                throw new NotImplementedException("(Sub)Variable Sizes not match");
            var result = new Variable();
            result.Weights = a.Weights - b.Weights;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            foreach (var item in b.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                a.Grads += result.Grads;
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                b.Grads -= result.Grads;
            };
            result.BackProp.Add(action);
            return result;
        }
        public static Variable Sub(Variable a, double b)
        {
            var result = new Variable();
            result.Weights = a.Weights - b;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                a.Grads += result.Grads;
            };
            result.BackProp.Add(action);
            return result;
        }
        public static Variable Sub(double a, Variable b)
        {
            var result = new Variable();
            result.Weights = a - b.Weights;
            foreach (var item in b.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            b.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                b.Grads -= result.Grads;
            };
            result.BackProp.Add(action);
            return result;
        }

        //行列の形変換
        public Variable Reshape(int row, int column)
        {
            var result = new Variable();
            result.Weights = Matrix.Reshape(this.Weights, row, column);
            foreach (var item in this.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            this.BackProp.Clear();
            Action action = () =>
              {
                  if (result.Grads == null)
                      result.Grads = OnesLike(result.Weights);
                  if (this.Grads == null)
                      this.Grads = MatClone(this.Weights);
                  this.Grads += Matrix.Reshape(result.Grads, this.Weights.Row, this.Weights.Column);
              };
            result.BackProp.Add(action);
            return result;
        }
        public Variable Transpose()
        {
            var result = new Variable();
            result.Weights = Matrix.MatTrans(this.Weights);
            foreach (var item in this.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            this.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (this.Grads == null)
                    this.Grads = MatClone(this.Weights);
                this.Grads = Matrix.MatTrans(result.Grads);
            };
            result.BackProp.Add(action);
            return result;
        }
        public Variable Sum()
        {
            var result = new Variable();
            result.Weights = Matrix.Sum(this.Weights);
            foreach (var item in this.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            this.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (this.Grads == null)
                    this.Grads = MatClone(this.Weights);
                this.Grads += Matrix.BloadCast(result.Grads, this.Weights.Row, this.Weights.Column); ;
            };
            result.BackProp.Add(action);
            return result;
        }
        public Variable Sum(int axis)
        {
            var result = new Variable();
            result.Weights = Matrix.Sum(this.Weights, axis);
            foreach (var item in this.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            this.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (this.Grads == null)
                    this.Grads = MatClone(this.Weights);

                this.Grads += Matrix.BloadCast(result.Grads, this.Weights.Row, this.Weights.Column); ;
            };
            result.BackProp.Add(action);
            return result;
        }
        /// <summary>
        /// 計算前に形を変形して行列計算をする関数
        /// </summary>
        /// <param name="a"></param>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public static Variable BloadCastTo(Variable a, int row, int column)
        {
            var result = new Variable();
            result.Weights = Matrix.BloadCast(a.Weights, row, column);
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                a.Grads += Matrix.SumTo(result.Grads, a.Weights.Row, a.Weights.Column);
            };
            result.BackProp.Add(action);
            return result;
        }

        /// <summary>
        ///　平均２乗誤差を求める関数
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Variable MeanSquaredError(Variable a, Variable b)
        {
            var result = new Variable();
            var diff = a.Weights - b.Weights;
            var y = Matrix.Sum(diff ^ 2) / (double)diff.Row;
            result.Weights = y;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            foreach (var item in b.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action action = () =>
              {
                  if (result.Grads == null)
                      result.Grads = OnesLike(result.Weights);
                  if (a.Grads == null)
                      a.Grads = MatClone(a.Weights);
                  if (b.Grads == null)
                      b.Grads = MatClone(b.Weights);
                  var diff = a.Weights - b.Weights;
                  var gy = BloadCast(result.Grads, a.Weights.Row, a.Weights.Column);
                  var gx = gy * diff * (2.0 / a.Weights.Row);
                  a.Grads += gx;
                  b.Grads -= gx;

              };
            result.BackProp.Add(action);
            return result;


        }

        public static Variable Linear(Variable x, Variable W, Variable b)
        {
            var result = new Variable();
            var y = MatDot(x.Weights, W.Weights);
            foreach (var action in x.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            foreach (var action in W.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            x.BackProp.Clear();
            W.BackProp.Clear();
            if (b != null)
            {
                y += b.Weights;
                foreach (var action in b.BackProp)
                {
                    if (!result.BackProp.Contains(action))
                        result.BackProp.Add(action);
                }
                b.BackProp.Clear();
            }
            Action backward = () =>
              {

                  if (result.Grads == null)
                      result.Grads = OnesLike(result.Weights);
                  if (x.Grads == null)
                      x.Grads = MatClone(x.Weights);
                  if (W.Grads == null)
                      W.Grads = MatClone(W.Weights);
                  if (b != null)
                  {
                      if (b.Grads == null)
                          b.Grads = MatClone(b.Weights);
                      b.Grads += SumTo(result.Grads, b.Weights.Row, b.Weights.Column);
                  }
                  x.Grads += MatDot(result.Grads, MatTrans(W.Weights));
                  W.Grads += MatDot(MatTrans(x.Weights), result.Grads);

              };
            result.Weights = y;
            result.BackProp.Add(backward);
            return result;
        }

        public static Variable SoftMaxCrossEntropy(Variable a, int[] b)
        {
            var result = new Variable();
            int N = a.Weights.Row;
            var logZ = LogSumExp(a.Weights, 1);
            var logP = a.Weights - logZ;
            var y = new Matrix(1, N);
            for (int i = 0; i < N; i++)
            {
                y.Data[0, i] = logP.Data[i, b[i]];
            }
            result.Weights = -Matrix.Sum(y) / (double)N;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }

            a.BackProp.Clear();

            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);

                int N = a.Weights.Row;
                int ClsNum = a.Weights.Column;
                result.Grads *= (double)1 / N;
                var y = SoftMax(a);
                var tOneHot = CreateMatrix(b.Count(), ClsNum);
                for (int i = 0; i < b.Count(); i++)
                {
                    tOneHot.Data[i, b[i]] = 1;
                }
                a.Grads += (y.Weights - tOneHot) * result.Grads;




            };
            result.BackProp.Add(action);
            return result;
        }
        public static Variable CrossEntropy(Variable a, int[] b)
        {
            var result = new Variable();
            int N = a.Weights.Row;
            var logZ = LogSumExp(a.Weights, 1);
            var logP = a.Weights - logZ;
            var y = new Matrix(1, N);
            for (int i = 0; i < N; i++)
            {
                y.Data[0, i] = logP.Data[i, b[i]];
            }
            result.Weights = -Matrix.Sum(y) / (double)N;
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }

            a.BackProp.Clear();

            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);

                int N = a.Weights.Row;
                int ClsNum = a.Weights.Column;
                result.Grads *= (double)1 / N;
                var y = SoftMax(a);
                var tOneHot = CreateMatrix(b.Count(), ClsNum);
                for (int i = 0; i < b.Count(); i++)
                {
                    tOneHot.Data[i, b[i]] = 1;
                }
                a.Grads += (y.Weights - tOneHot) * result.Grads;




            };
            result.BackProp.Add(action);
            return result;
        }
        public static Variable CrossEntropy(Variable a, int b)
        {
            var result = new Variable();
            int N = a.Weights.Row;
            var logZ = LogSumExp(a.Weights, 1);
            var logP = a.Weights - logZ;
            var y = new Matrix(1, 1);
                y.Data[0, 0] = logP.Data[0, b];
            result.Weights = -Matrix.Sum(y);
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }

            a.BackProp.Clear();

            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);

                int N = a.Weights.Row;
                int ClsNum = a.Weights.Column;
                var y = SoftMax(a);
                var tOneHot = CreateMatrix(1, ClsNum);
                    tOneHot.Data[0, b] = 1;
                a.Grads += (y.Weights - tOneHot) * result.Grads;




            };
            result.BackProp.Add(action);
            return result;
        }

        /*Activation function */
        public static Variable Sigmoid(Variable x)
        {
            var result = new Variable();
            result.Weights = MatTanh(x.Weights * 0.5) * 0.5 + 0.5;
            foreach (var act in x.BackProp)
            {
                if (!result.BackProp.Contains(act))
                    result.BackProp.Add(act);
            }
            x.BackProp.Clear();
            Action action = () =>
             {
                 if (result.Grads == null)
                     result.Grads = OnesLike(result.Weights);
                 if (x.Grads == null)
                     x.Grads = MatClone(x.Weights);
                 x.Grads += result.Grads * result.Weights * (1.0 - result.Weights);
             };
            result.BackProp.Add(action);
            return result;
        }

        public static Variable ReLu(Variable a)
        {
            var result = Clone(a);
            result.Weights = Maximum(a.Weights, 0.0);
            foreach (var item in a.BackProp)
            {
                if (!a.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                var mask = Maximum(result.Grads, 0.0);
                a.Grads += result.Grads * mask;
            };
            result.BackProp.Add(action);
            return result;
        }

        public static Variable SoftMax(Variable a, int? axis = 1)
        {
            var result = Clone(a);
            var y = a.Weights - Max(a.Weights, axis);
            y = MatExp(y);
            result.Weights = y / Matrix.Sum(y, axis);
            foreach (var item in a.BackProp)
            {
                if (!a.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                var gx = result.Weights * result.Grads;
                var sumgx = Matrix.Sum(gx, axis);
                a.Grads -= result.Weights * sumgx;
            };
            result.BackProp.Add(action);
            return result;
        }

        public static Variable Tanh(Variable x)
        {
            var result = new Variable();
            result.Weights = MatTanh(x.Weights);
            foreach (var act in x.BackProp)
            {
                if (!result.BackProp.Contains(act))
                    result.BackProp.Add(act);
            }
            x.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike(result.Weights);
                if (x.Grads == null)
                    x.Grads = MatClone(x.Weights);
                x.Grads += result.Grads * (1.0 - result.Weights * result.Weights);
            };
            result.BackProp.Add(action);
            return result;
        }
        /// <summary>
        /// aの要素をmin以上max以下に変換する
        /// </summary>
        /// <param name="a"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static Variable Clip(Variable a, double min, double max)
        {
            Variable result = new Variable();
            result.Weights = Minimum(Maximum(a.Weights, min), max);
            foreach (var item in a.BackProp)
            {
                if (!result.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
              {
                  if (result.Grads == null)
                      result.Grads = OnesLike(result.Weights);
                  if (a.Grads == null)
                      a.Grads = MatClone(a.Weights);
                  var mask = (a.Weights >= max) * (a.Weights <= min);
                  a.Grads += result.Grads * mask;
              };
            result.BackProp.Add(action);
            return result;
        }
        /// <summary>
        /// embeddingから単語ベクトルを取り出す関数
        /// </summary>
        /// <param name="x">Embedding Matrix</param>
        /// <param name="ix">wordnumber</param>
        /// <returns></returns>
        public static Variable PeekRow(Variable x, int ix)
        {
            var d = x.Column;
            var res = new Variable(CreateMatrix(1,d));
            for (int i = 0; i < d; i++)
            {
                res.Weights.Data[0, i] = x.Weights.Data[ix, i];
            }
            foreach (var item in x.BackProp)
            {
                if (!res.BackProp.Contains(item))
                    res.BackProp.Add(item);
            }
            x.BackProp.Clear();
            Action action = () =>
            {
                if (x.Grads == null)
                    x.Grads = MatClone(x.Weights);
                for (int i = 0; i < d; i++)
                {
                    x.Grads.Data[ix, i] += res.Grads.Data[0, i];
                }
            };
            res.BackProp.Add(action);
            return res;


        }
        /// <summary>
        /// 文章のベクトルを取り出す関数
        /// </summary>
        /// <param name="x">Embedding Matrix</param>
        /// <param name="ix">TextNumbers</param>
        /// <returns></returns>
        public static List<Variable> PeekRow(Variable x, int[] ix)
        {
            var d = x.Column;
            var res = new List<Variable>();
            var y = new Matrix(ix.Length, d);
            for (int j = 0; j < ix.Length; j++) {
                    res.Add(PeekRow(x, ix[j]));
            }
          
            return res;


        }
        /// <summary>
        /// ２つの行列を横並びでつなぐ
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Variable ConcatColumn(Variable a, Variable b)
        {
            Variable result = new Variable();
            result.Weights = new Matrix(a.Weights.Row, a.Weights.Column + b.Weights.Column);
            for (int i = 0; i < a.Weights.Row; i++)
            {
                for (int j = 0; j < a.Weights.Column; j++)
                {
                    result.Weights.Data[i, j] = a.Weights.Data[i, j];
                }
                for (int j = 0; j < b.Weights.Column; j++)
                {
                    result.Weights.Data[i, j+a.Column] = b.Weights.Data[i, j];
                }
            }
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            foreach (var action in b.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action backward = () => 
            {
                if (result.Grads == null)
                    result.Grads = ZerosLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                for (int i = 0; i < a.Weights.Row; i++)
                {
                  
                    for (int j = 0; j < a.Weights.Column; j++)
                    {
                         a.Grads.Data[i, j]+= result.Grads.Data[i, j] ;
                    }
                    for (int j = 0; j < b.Weights.Column; j++)
                    {
                        b.Grads.Data[i, j] += result.Grads.Data[i, j + a.Column];
                    }

                }
                
            };
            result.BackProp.Add(backward);
            return result;

        }
        /// <summary>
        /// ２つの行列を縦並びでつなぐ
        /// </summary>
        /// <param name="a">Variable</param>
        /// <param name="b">Variable</param>
        /// <returns>Variable</returns>
        public static Variable ConcatRow(Variable a, Variable b)
        {
            Variable result = new Variable();
            result.Weights = new Matrix(a.Weights.Row+b.Weights.Row, a.Weights.Column );
            
            
                for (int i = 0; i < a.Weights.Row; i++)
                {
                    for (int j = 0; j < a.Weights.Column; j++)
                        result.Weights.Data[i, j] = a.Weights.Data[i, j];
                }
                for (int i = 0; i < b.Weights.Row; i++)
                {
                    for (int j = 0; j < a.Weights.Column; j++)
                        result.Weights.Data[i+a.Weights.Row, j ] = b.Weights.Data[i, j];
                }
            
            foreach (var action in a.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            foreach (var action in b.BackProp)
            {
                if (!result.BackProp.Contains(action))
                    result.BackProp.Add(action);
            }
            a.BackProp.Clear();
            b.BackProp.Clear();
            Action backward = () =>
            {
                if (result.Grads == null)
                    result.Grads = ZerosLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
            
                    for (int i = 0; i < a.Weights.Row; i++)
                    {
                    for (int j = 0; j < a.Weights.Column; j++)
                        a.Grads.Data[i, j] += result.Grads.Data[i, j];
                    }
                    for (int i = 0; i < b.Weights.Row; i++)
                    {
                    for (int j = 0; j < a.Weights.Column; j++)
                        b.Grads.Data[i, j] += result.Grads.Data[i+a.Weights.Row, j];
                    }


            };
            result.BackProp.Add(backward);
            return result;

        }
        public static Variable ScaleMul(Variable a, Variable b)
        {
            var result = new Variable(a.Weights);
            foreach (var act in a.BackProp)
            {
                if (!result.BackProp.Contains(act))
                    result.BackProp.Add(act);
            }
            a.BackProp.Clear();
            foreach (var act in b.BackProp)
            {
                if (!result.BackProp.Contains(act))
                    result.BackProp.Add(act);
            }
            b.BackProp.Clear();
            for (int i = 0; i < a.Row; i++)
                for (int j = 0; j < a.Column; j++)
                    result.Weights.Data[i, j] = a.Weights.Data[i, j] * b.Weights.Data[0, 0];
            Action backward = () =>
            {
                if (result.Grads == null)
                    result.Grads = ZerosLike(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone(a.Weights);
                if (b.Grads == null)
                    b.Grads = MatClone(b.Weights);
                for (int i = 0; i < a.Weights.Row; i++)
                {

                    for (int j = 0; j < a.Weights.Column; j++)
                    {
                        a.Grads.Data[i, j] +=b.Weights.Data[0,0]* result.Grads.Data[i, j];
                        b.Grads.Data[0, 0] += a.Weights.Data[i, j] * result.Grads.Data[i, j];
                    }

                }

            };
            result.BackProp.Add(backward);

            return result;
        }
    }
    public class Parameter : Variable
    {
        public Parameter()
        { }
        public Parameter(Matrix a)
        {
            Weights = a;
            this.Row = a.Row;
            this.Column = a.Column;
        }
        public Parameter(double a)
        {
            Weights = new Matrix(a);
        }
        public Parameter(double[,] a)
        {
            Weights = new Matrix(a);
        }
    }
    /// <summary>
    /// ４次元データを扱えるVariableClass
    /// </summary>
    public class Variable4D
    {
        public Matrix[,] Weights;
        public Matrix[,] Grads;
        public int N;
        public int C;
        public int Row;
        public int Column;
        public List<Action> BackProp = new List<Action>();
        public Variable4D()
        { }
        public  Variable4D(Matrix[,] x)
        {
            this.Weights = x;
            this.N = x.GetLength(0);
            this.C = x.GetLength(1);
            this.Row = x[0, 0].Row;
            this.Column = x[0, 0].Column;
        }
        public (int, int, int, int) Shape()
        {
            return (N, C, Row, Column);
        }
        public static Variable4D ReLu4D(Variable4D a)
        {
            var result = Clone4D(a);
            for (int i = 0; i < result.N; i++)
            {
                for(int j=0;j<result.C;j++)
                    result.Weights[i,j] = Maximum(a.Weights[i,j], 0.0);
            }
            foreach (var item in a.BackProp)
            {
                if (!a.BackProp.Contains(item))
                    result.BackProp.Add(item);
            }
            a.BackProp.Clear();
            Action action = () =>
            {
                if (result.Grads == null)
                    result.Grads = OnesLike4D(result.Weights);
                if (a.Grads == null)
                    a.Grads = MatClone4D(a.Weights);
                for (int i = 0; i < result.N; i++)
                {
                    for (int j = 0; j < result.C; j++)
                    {
                        var mask = Maximum(result.Grads[i,j], 0.0);
                        a.Grads[i,j] += result.Grads[i,j] * mask;
                    }
                }
              
            };
            result.BackProp.Add(action);
            return result;
        }
        public static Variable4D Clone4D(Variable4D a)
        {
            var result =new Matrix[a.N,a.C];
            for (int i = 0; i < a.N; i++)
            {
                for (int j = 0; j < a.C; j++)
                    result[i, j] = MatClone(a.Weights[0, 0]);
            }
            return new Variable4D(result);
        }
        public static Matrix[,] MatClone4D(Matrix[,] a)
        {
            var result = new Matrix[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                    result[i, j] = MatClone(a[0, 0]);
            }
            return result;
        }
        public static Matrix[,] OnesLike4D(Matrix[,] a)
        {
            var result = new Matrix[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                    result[i, j] = OnesLike(a[0, 0]);
            }
            return result;
        }
        /// <summary>
        /// 行列の形を変える関数
        /// </summary>
        /// <param name="mat"></param>
        /// <param name="a～d">0～3（重複なし）</param>
        /// <returns></returns>
        public static Matrix[,] Reshape(Matrix[,] mat, int a, int b, int c1, int d)
        {
            
            int n = mat.GetLength(0);
            int c = mat.GetLength(1);
            int h = mat[0,0].Row;
            int w = mat[0,0].Column;
            int ax = c * h * w;
            int bx = h * w;
            int c1x = w;
            //係数
            int nx = 1;
            int cx = 1;
            int hx = 1;
            int wx = 1;
            //大きさ
            int nmax = n;
            int cmax = c;
            int hmax = h;
            int wmax = w;
            

            var x = new double[n * c * h * w];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int k = 0; k < h; k++)
                    {
                        for (int l = 0; l < w; l++)
                        {
                            x[(i * c + j) * h * w + k * w + l] = mat[i, j].Data[k, l];
                        }
                    }
                }
            }
            switch (a)
            {
                case 0:
                    nx = ax;
                    break;
                case 1:
                    nx = bx;
                    nmax = c;
                    break;
                case 2:
                    nx = c1x;
                    nmax = h;
                    break;
                case 3:
                    nx = 1;
                    nmax = w;
                    break;
            }
            switch (b)
            {
                case 0:
                    cx = ax;
                    cmax = n;
                    break;
                case 1:
                    cx = bx;
                    cmax = c;
                    break;
                case 2:
                    cx = c1x;
                    cmax = h;
                    break;
                case 3:
                    cx = 1;
                    cmax = w;
                    break;
            }
            switch (c)
            {
                case 0:
                    hx = ax;
                    hmax = n;
                    break;
                case 1:
                    hx = bx;
                    hmax = c;
                    break;
                case 2:
                    hx = c1x;
                    hmax = h;
                    break;
                case 3:
                    hx = 1;
                    hmax = w;
                    break;
            }
            switch (d)
            {
                case 0:
                    wx = ax;
                    wmax = n;
                    break;
                case 1:
                    wx = bx;
                    wmax = c;
                    break;
                case 2:
                    wx = c1x;
                    wmax = h;
                    break;
                case 3:
                    wx = 1;
                    wmax = w;
                    break;
            }
            var result = new Matrix[nmax, cmax];
            for (int i = 0; i <nmax ; i++)
            {
                for (int j = 0; j < cmax; j++)
                {
                    var mat1 = new Matrix(hmax, wmax);
                    for (int k = 0; k < hmax; k++)
                    {
                        
                        for (int l = 0; l < wmax; l++)
                        {
                            mat1.Data[k, l] = x[i*nx+j*cx+k*hx+l*wx]; 
                        }
                        
                    }
                    result[i, j] = mat1;
                }
            }
            return result;

        }
        
    }

   
}
