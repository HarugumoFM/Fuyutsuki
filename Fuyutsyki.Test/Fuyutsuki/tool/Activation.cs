using System;
using static Fuyutsuki.Variable;

namespace Fuyutsuki.tool
{
    /// <summary>
    /// 活性化関数のクラス
    /// </summary>
    public class Activations
    {
        public class Activation
        {
            public virtual Variable Forward(Variable x)
            {
                throw new NotImplementedException();
            }
        }
        public class Sigmoid : Activation
        {
            public override Variable Forward(Variable x)
            {
                return Sigmoid(x);
            }
        }
        public class Relu : Activation
        {
            public override Variable Forward(Variable x)
            {
                return ReLu(x);
            }
        }
        public class SoftMax:Activation
        {
            public int? Axis;
            public SoftMax(int? axis=1)
            {
                Axis = axis;
            }
            public override Variable Forward(Variable x)
            {
                return SoftMax(x,Axis);
            }

        }
    }
}
