using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using static Fuyutsuki.Matrix;
using static Fuyutsuki.Variable;
namespace Fuyutsuki
{
    public class Function
    {
        public  Variable Reshape(Variable array, int row,int column)
        {
            if (array.Weights.Row == row && array.Weights.Column == column)
                return array;
            else
                return array.Reshape(row, column);

        }
        public  Variable Transpose(Variable x)
        {
            return x.Transpose();
        }
        public  Variable MeanSquaredError(Variable x0, Variable x1)
        {
            return Variable.MeanSquaredError(x0, x1);
        }
        public Variable MatMul(Variable a, Variable b)
        {
            return Dot(a, b);
        }
        public Variable Linear(Variable x, Variable W, Variable b = null)
        {
            return Variable.Linear(x, W, b);
        }
        public Variable Tanh(Variable x)
        {
           return Variable.Tanh(x);
        }
        //Variableのデータを縦方向に結合
        public Variable Concat(List<Variable> xs)
        {
            int row = xs.Count();
            int column = xs[0].Weights.Column;
            var result = new Variable();
            result.Weights = CreateMatrix(row, column);
            for(int i=0;i<row;i++)
            {
                for (int j = 0; j < column; j++)
                    result.Weights.Data[i, j] = xs[i].Weights.Data[0, j];
                foreach (var action in xs[i].BackProp)
                {
                    if (!result.BackProp.Contains(action))
                        result.BackProp.Add(action);
                    
                }
                xs[i].BackProp.Clear();

            }
            Action action1 = () =>
            {
                if (result.Grads == null)
                    result.Grads = ZerosLike(result.Weights);
                for (int i = 0; i < row; i++)
                {
                    if (xs[i].Grads == null)
                        xs[i].Grads = MatClone(xs[i].Weights);
                    for (int j = 0; j < row; j++)
                        xs[i].Grads.Data[0, j] += result.Grads.Data[i, j];
                }
            };
            result.BackProp.Add(action1);
            return result;
        }
        public Variable Sigmoid(Variable x)
        {
            return Variable.Sigmoid(x);
        }
        /// <summary>
        /// 0～num-1までの番号がシャッフルされた配列を返す関数
        /// </summary>
        /// <param name="num"></param>
        /// <returns></returns>
        public int[] Permutation(int num)
        {
            var result = new int[num];
            for (int i = 0; i < num; i++)
            {
                result[i] = i;
            }
            return result.OrderBy(nt => Guid.NewGuid()).ToArray();
        }
        /// <summary>
        /// 0～num-1までの番号の配列を返す
        /// </summary>
        /// <param name="num"></param>
        /// <returns></returns>
        public int[] Arrange(int num)
        {
            var result = new int[num];
            for (int i = 0; i < num; i++)
            {
                result[i] = i;
            }
            return result;
        }
        /// <summary>
        /// xからyのデータを抽出する
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public int[] Squeeze(int[] x, int[] y)
        {
            var result = new int[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                result[i] = x[y[i]];
            }
            return result;
        }
        /// <summary>
        /// 特定のVariableをドロップアウトさせる関数
        /// </summary>
        /// <param name="x">Dropoutする行列</param>
        /// <param name="train">学習中か否か</param>
        /// <param name="dropoutRatio">ドロップ率</param>
        /// <returns></returns>
        public  Variable DropOut(Variable x,bool train,double dropoutRatio=0.5)
        {
            if (train)
            {
                var mask = RandN(x.Row, x.Column) > dropoutRatio;
                return x * new Variable(mask) / (1.0 - dropoutRatio);
            }
            else
                return x;
        }
        

       
    }
}
