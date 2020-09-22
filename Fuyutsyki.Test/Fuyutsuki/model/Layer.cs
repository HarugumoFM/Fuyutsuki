using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Fuyutsuki.Variable;
using static Fuyutsuki.Matrix;
using System.ComponentModel.Design;
using System.Security.Cryptography.X509Certificates;

namespace Fuyutsuki.model
{
    public class Layers
    {
        /// <summary>
        /// レイヤー・モデルの基底クラス
        /// </summary>
        public class Layer
        {
            public List<Parameter> Params;
            public Function F = new Function();
            public List<Layer> Layers;
            public Variable Outputs;
            public Variable inputs;

            public Layer()
            {
                Params = new List<Parameter>();
                Layers = new List<Layer>();
            }
            public Variable Call(Variable input)
            {
                var outputs = Forward(input);
                inputs = input;
                Outputs = outputs;
                return outputs;
            }
            public virtual Variable Forward(Variable inputs)
            {
                throw new NotImplementedException();
            }
            public virtual Variable Forward(List<Variable> inputs)
            {
                throw new NotImplementedException();
            }
            
            public virtual Variable4D Forward(Variable4D inputs)
            {
                throw new NotImplementedException();
            }
            /// <summary>
            /// パラメータを返す関数
            /// </summary>
            /// <returns></returns>
            public virtual IEnumerable<Parameter> GetParam()
            {
                foreach (var param in Params)
                {
                    yield return param;
                }
                //内部レイヤーが存在する場合そのパラメーターも返す
                if (Layers.Count > 0)
                    foreach (var layer in Layers)
                    {
                        foreach (var param in layer.GetParam())
                        {
                            yield return param;
                        }
                    }
            }
            public virtual void ClearGrads()
            {
                foreach (var param in Params)
                {
                    param.ClaerGrads();
                }
                if (Layers.Count > 0)
                {
                    foreach (var layer in Layers)
                    {
                        layer.ClearGrads();
                    }
                }
            }
            public void SettingsBindableAttribute(string name, object obj)
            {
                if (obj.GetType() == typeof(Parameter))
                    Params.Add((Parameter)obj);

            }
            public void AddLayer(Layer layer)
            {
                Layers.Add(layer);
                
            }
        }
        /// <summary>
        /// Dense(Linear) Layer
        /// </summary>
        public class Linear : Layer
        {
            int I = 0;
            int O = 0;
            Parameter W;
            Parameter b;
            /// <summary>
            /// 
            /// </summary>
            /// <param name="inSize">入力次元</param>
            /// <param name="outSize">出力次元</param>
            /// <param name="noBias">バイアスの有無</param>
            public Linear(int? inSize, int outSize, bool noBias = false)
            {
                if (inSize == null)
                {
                    O = outSize;
                    if (!noBias)
                    {
                        b = new Parameter(new Matrix(1,outSize));
                        Params.Add(b);
                    }
                }
                else
                {
                    I = (int)inSize;
                    O = outSize;
                    var WData = RandN(I, O) * Math.Sqrt((double)1 / I);
                    W = new Parameter(WData);
                    Params.Add(W);
                    if (!noBias)
                    {
                        b = new Parameter(new Matrix(0));
                        Params.Add(b);
                    }
                }
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="inSize">入力サイズ</param>
            /// <param name="outSize">出力サイズ</param>
            /// <param name="maskWeight">重みの係数</param>
            /// <param name="noBias">バイアスの有無</param>
            public Linear(int? inSize, int outSize, int maskWeight,bool noBias = false)
            {
                if (inSize == null)
                {
                    O = outSize;
                    if (!noBias)
                    {
                        b = new Parameter(new Matrix(1, outSize));
                        Params.Add(b);
                    }
                }
                else
                {
                    I = (int)inSize;
                    O = outSize;
                    var WData = RandN(I, O) * maskWeight;
                    W = new Parameter(WData);
                    Params.Add(W);
                    if (!noBias)
                    {
                        b = new Parameter(new Matrix(0));
                        Params.Add(b);
                    }
                }
            }
            public Linear(int outSize, bool noBias = false)
            {

                O = outSize;
                if (!noBias)
                {
                    b = new Parameter(new Matrix(1,outSize));
                    Params.Add(b);
                }
            }
            public override Variable Forward(Variable inputs)
            {
                if (W == null)
                {
                    this.I = inputs.Weights.Column;
                    double n = Math.Sqrt((double)1 / I);
                    W = new Parameter(RandN(I, O) * n);
                    Params.Add(W);

                }
                var y = F.Linear(inputs, W, b);
                return y;
            }

        }
        [Serializable]
        /// <summary>
        /// 単語等をEmbeddするLayer
        /// </summary>
        public class Embedding : Layer
        {
            Parameter Embeddings;
            /// <summary>
            /// 
            /// </summary>
            /// <param name="vocabSize">ワード数</param>
            /// <param name="embeddingDim">embeddingの次元</param>
            /// <param name="Padidx">パディング番号</param>
            public Embedding(int vocabSize,int embeddingDim,int? Padidx=null)
            {
                Embeddings = new Parameter(RandN(vocabSize, embeddingDim));
                if (Padidx != null)
                    for (int i = 0; i < embeddingDim; i++)
                    {
                        Embeddings.Weights.Data[(int)Padidx, i] = 0.0;
                    }
                Params.Add(Embeddings);
            }
            /// <summary>
            /// wordIDのVectorを抽出する関数
            /// </summary>
            /// <param name="ix">ID</param>
            /// <returns>Vector</returns>
            public Variable Embed(int ix)
            {
                return PeekRow(Embeddings, ix);
            }
            /// <summary>
            /// IDのリストからベクトルのリストを返す関数
            /// </summary>
            /// <param name="ix">IDのリスト</param>
            /// <returns></returns>
            public List<Variable> Embed(int[] ix)
            {
                return PeekRow(Embeddings, ix);
            }
        }




        /*Recurrent Neural Network*/
        public class RNNCell : Layer
        {
            Layer x2h;
            Layer h2h;
            public Variable h;
            public RNNCell(int hiddenSize, int? inSize = null)
            {
                x2h = new Linear(inSize, hiddenSize);
                h2h = new Linear(inSize, hiddenSize, true);
                Layers.Add(x2h);
                Layers.Add(h2h);
                h = null;



            }
            public virtual void ResetState()
            {
                h = null;
            }
           
            public  virtual Variable Forward(Variable inputs,Variable oldH=null)
            {

                if (oldH == null)
                    h = F.Tanh(x2h.Forward(inputs));
                else
                    h = F.Tanh(x2h.Forward(inputs) + h2h.Forward(oldH));
                return h;

            }
        }
        public class RNN : Layer
        {
            int SeqLen;
            public List<RNNCell> RNNLayers = new List<RNNCell>();
            /// <summary>
            /// GRUレイヤーのコンストラクタ
            /// </summary>
            /// <param name="hiddenSize">(int)隠れ層の次元</param>
            /// <param name="seqlen">(int)系列の長さ</param>
            /// <param name="inSize"></param>
            public RNN(int hiddenSize, int seqlen, int? inSize = null)
            {
                this.SeqLen = seqlen;
                for (int i = 0; i < seqlen; i++)
                {
                    RNNLayers.Add(new RNNCell(hiddenSize, inSize));
                }
            }
            /// <summary>
            /// 時系列データを流す
            /// </summary>
            /// <param name="inputs">時系列データ</param>
            /// <returns>h:一番最後の層の出力 outputs:すべての層の出力</returns>
            public (Variable h, List<Variable> outputs) Forward(List<Variable> inputs, Variable oldH = null)
            {
                var output = oldH;
                List<Variable> outputs = new List<Variable>();
                for (int i = 0; i < this.SeqLen; i++)
                {
                    var output1 = RNNLayers[i].Forward(inputs[i], output);
                    output = null;
                    output = output1;
                    outputs.Add(output);
                }
                return (output, outputs);
            }
            public (Variable h, List<Variable> outputs) Forward(Variable inputs, Variable oldH = null)
            {
                var output = oldH;
                var input = inputs;
                List<Variable> outputs = new List<Variable>();
                for (int i = 0; i < SeqLen; i++)
                {
                    input = RNNLayers[i].Forward(input, output);
                    output = null;
                    output = input;
                    outputs.Add(output);
                }
                return (output, outputs);
            }



            public void ResetStatu()
            {
                RNNLayers[0].h = null;
            }
            public override IEnumerable<Parameter> GetParam()
            {
                foreach (var layer in RNNLayers)
                {
                    foreach (var param in layer.GetParam())
                        yield return param;
                }
                foreach (var param in base.GetParam())
                    yield return param;
            }

        }


        /// <summary>
        /// LSTMネットワークを形成するLSTMCell
        /// </summary>
        public class LSTMCell : RNNCell
        {
            Linear x2F;
            Linear x2I;
            Linear x2O;
            Linear x2U;
            Linear h2F;
            Linear h2I;
            Linear h2O;
            Linear h2U;
            public Variable c;

            public LSTMCell(int hiddenSize,int? inSize=null):base(hiddenSize,inSize)
            {
                x2F = new Linear(inSize,hiddenSize);
                x2I = new Linear(inSize, hiddenSize);
                x2O = new Linear(inSize, hiddenSize);
                x2U = new Linear(inSize, hiddenSize);
                h2F = new Linear(hiddenSize, hiddenSize, true);
                h2I = new Linear(hiddenSize, hiddenSize, true);
                h2O = new Linear(hiddenSize, hiddenSize, true);
                h2U = new Linear(hiddenSize, hiddenSize, true);
                Layers.Add(x2F);
                Layers.Add(x2I);
                Layers.Add(x2O);
                Layers.Add(x2U);
                Layers.Add(h2F);
                Layers.Add(h2I);
                Layers.Add(h2O);
                Layers.Add(h2U);
                this.ResetState();
            }
            public override void ResetState()
            {
                this.h = null;
                this.c = null;
            }
            public override Variable Forward(Variable inputs)
            {
                Variable f;
                Variable i;
                Variable o;
                Variable u;

                if (h == null)
                {
                     f = F.Sigmoid(x2F.Forward(inputs));
                     i = F.Sigmoid(x2I.Forward(inputs));
                     o = F.Sigmoid(x2O.Forward(inputs));
                     u = F.Sigmoid(x2U.Forward(inputs));
                }
                else
                {
                    f = F.Sigmoid(x2F.Forward(inputs)+h2F.Forward(h));
                    i = F.Sigmoid(x2I.Forward(inputs)+h2I.Forward(h));
                    o = F.Sigmoid(x2O.Forward(inputs)+h2O.Forward(h));
                    u = F.Sigmoid(x2U.Forward(inputs)+h2U.Forward(h));
                }
                if (c == null)
                    c = i * u;
                else
                    c = (f * c) + (i * u);
                h = o + F.Tanh(c);
                return h;
            }
            public  (Variable h,Variable c) Forward(Variable inputs,Variable oldh,Variable oldc)
            {
                Variable f;
                Variable i;
                Variable o;
                Variable u;

                if (oldh == null)
                {
                    f = F.Sigmoid(x2F.Forward(inputs));
                    i = F.Sigmoid(x2I.Forward(inputs));
                    o = F.Sigmoid(x2O.Forward(inputs));
                    u = F.Sigmoid(x2U.Forward(inputs));
                }
                else
                {
                    f = F.Sigmoid(x2F.Forward(inputs) + h2F.Forward(oldh));
                    i = F.Sigmoid(x2I.Forward(inputs) + h2I.Forward(oldh));
                    o = F.Sigmoid(x2O.Forward(inputs) + h2O.Forward(oldh));
                    u = F.Sigmoid(x2U.Forward(inputs) + h2U.Forward(oldh));
                }
                if (oldc == null)
                    c = i * u;
                else
                    c = (f * oldc) + (i * u);
                h = o + F.Tanh(c);
                return (h,c);
            }

        }
        public class LSTM : Layer
        {
            int SeqLen;
            public List<LSTMCell> LSTMLayers = new List<LSTMCell>();
            /// <summary>
            /// GRUレイヤーのコンストラクタ
            /// </summary>
            /// <param name="hiddenSize">(int)隠れ層の次元</param>
            /// <param name="Seqlen">(int)系列の長さ</param>
            /// <param name="inSize"></param>
            public LSTM(int hiddenSize, int Seqlen, int? inSize = null)
            {
                SeqLen = Seqlen;
                for (int i = 0; i < Seqlen; i++)
                {
                    LSTMLayers.Add(new LSTMCell(hiddenSize, inSize));
                }
            }
            /// <summary>
            /// 時系列データを流す
            /// </summary>
            /// <param name="inputs">時系列データ</param>
            /// <returns>h:一番最後の層の出力 outputs:すべての層の出力</returns>
            public (Variable h, List<Variable> outputs) Forward(List<Variable> inputs, Variable oldH = null,Variable oldC=null)
            {
                var OldH = oldH;
                var OldC = oldC;
                List<Variable> outputs = new List<Variable>();
                for (int i = 0; i < SeqLen; i++)
                {
                    (OldH,OldC)= LSTMLayers[i].Forward(inputs[i], OldH,OldC);
                    outputs.Add(OldH);
                }
                return (OldH, outputs);
            }
            public (Variable h, List<Variable> outputs) Forward(Variable inputs, Variable oldH = null, Variable oldC = null)
            {
                var OldH = oldH;
                var OldC = oldC;
                var input = inputs;
                List<Variable> outputs = new List<Variable>();
                for (int i = 0; i < SeqLen; i++)
                {
                    (OldH, OldC) = LSTMLayers[i].Forward(input, OldH, OldC);
                    input = oldH;
                    outputs.Add(OldH);
                }
                return (OldH, outputs);
            }



            public void ResetStatu()
            {
                LSTMLayers[0].h = null;
                LSTMLayers[0].h = null;
            }
            public override IEnumerable<Parameter> GetParam()
            {
                foreach (var layer in LSTMLayers)
                {
                    foreach (var param in layer.GetParam())
                        yield return param;
                }
                foreach (var param in base.GetParam())
                    yield return param;
            }

        }


        [Serializable]
        /// <summary>
        /// GRUネットワークを形成するGRUCell
        /// </summary>
        public class GRUCell : RNNCell
        {
            protected Linear x2Z;
            protected Linear x2R;
            protected Linear x2H;
            protected Linear h2Z;
            protected Linear h2R;
            protected Linear h2H;

            public GRUCell(int hiddenSize, int? inSize = null) : base(hiddenSize, inSize)
            {
                x2Z = new Linear(inSize, hiddenSize);
                x2R = new Linear(inSize, hiddenSize);
                x2H = new Linear(inSize, hiddenSize);
                h2Z = new Linear(hiddenSize, hiddenSize,true);
                h2R = new Linear(hiddenSize, hiddenSize,true);
                h2H = new Linear(hiddenSize, hiddenSize,true);
                Layers.Add(x2Z);
                Layers.Add(x2R);
                Layers.Add(x2H);
                Layers.Add(h2Z);
                Layers.Add(h2R);
                Layers.Add(h2H);
                this.ResetState();
            }
            public override void ResetState()
            {
                this.h = null;
            }
            public override Variable Forward(Variable inputs)
            {
                Variable z;
                Variable r;
                
                

                if (h == null)
                {
                    z = F.Sigmoid(x2Z.Forward(inputs));
                    r = F.Sigmoid(x2R.Forward(inputs));
                    h = z*F.Tanh(x2H.Forward(inputs));
                }
                else
                {
                    z = F.Sigmoid(x2Z.Forward(inputs)+h2Z.Forward(h));
                    r = F.Sigmoid(x2R.Forward(inputs)+h2R.Forward(h));
                    h = ((1.0 - z)* h) + 
                        z*F.Tanh(x2H.Forward(inputs)+h2H.Forward(r*h));
                }
               
               
                return h;
            }
            /// <summary>
            /// RNNcellを進める関数
            /// </summary>
            /// <param name="inputs">入力</param>
            /// <param name="oldH">一つ前の層の隠れ層</param>
            /// <returns></returns>
            public override Variable Forward(Variable inputs,Variable oldH)
            {
                if (oldH == null)
                    return Forward(inputs);
                else
                {
                    var r1 = x2Z.Forward(inputs) + h2Z.Forward(oldH);
                    var z = F.Sigmoid(r1);
                    var r = F.Sigmoid(x2R.Forward(inputs) + h2R.Forward(oldH));
                    this.h = ((1.0 - z) * oldH) +
                          z * F.Tanh(x2H.Forward(inputs) + h2H.Forward(r * oldH));



                    return h;
                }
            }

        }
        [Serializable]
        public class ContextGRUCell : GRUCell
        {
            Linear c2Z;
            Linear c2R;
            Linear c2H;
            int ContextSize;
            public ContextGRUCell(int hiddenSize,int contextSize,int? insize):base(hiddenSize,insize)
            {
                this.ContextSize = contextSize;
                c2Z = new Linear(ContextSize, hiddenSize);
                c2R = new Linear(ContextSize, hiddenSize);
                c2H = new Linear(ContextSize, hiddenSize);
                Layers.Add(c2Z);
                Layers.Add(c2R);
                Layers.Add(c2H);
               
            }
            public override void ResetState()
            {
                this.h = null;
            }
            public  new Variable Forward(Variable inputs,Variable Context)
            {
                Variable z;
                Variable r;



                if (h == null)
                {
                    z = F.Sigmoid(x2Z.Forward(inputs)+c2Z.Forward(Context));
                    r = F.Sigmoid(x2R.Forward(inputs)+c2R.Forward(Context));
                    h = z * F.Tanh(x2H.Forward(inputs)+Dot(r,c2H.Forward(Context)));
                }
                else
                {
                    z = F.Sigmoid(x2Z.Forward(inputs) + h2Z.Forward(h)+c2Z.Forward(Context));
                    r = F.Sigmoid(x2R.Forward(inputs) + h2R.Forward(h)+c2R.Forward(Context));
                    h = ((1.0 - z) * h) +
                        z * F.Tanh(x2H.Forward(inputs) + 
                        Dot(r, (h2H.Forward(h) + c2H.Forward(Context)) )
                        );
                }


                return h;
            }
            /// <summary>
            /// RNNcellを進める関数
            /// </summary>
            /// <param name="inputs">入力</param>
            /// <param name="oldH">一つ前の層の隠れ層</param>
            /// <returns></returns>
            public  Variable Forward(Variable inputs, Variable Context,Variable oldH)
            {
                if (oldH == null)
                    return Forward(inputs);
                else
                {
                    var r1 = x2Z.Forward(inputs) + h2Z.Forward(oldH)+c2Z.Forward(Context);
                    var z = F.Sigmoid(r1);
                    var r = F.Sigmoid(x2R.Forward(inputs) + h2R.Forward(oldH)+c2R.Forward(Context));
                    this.h = ((1.0 - z) * oldH) +
                          z * F.Tanh(x2H.Forward(inputs) + Dot(r,h2H.Forward(oldH)+c2H.Forward(Context)));



                    return h;
                }
            }
        }
        public class GRU : Layer
        {

            List<GRUCell> gru = new List<GRUCell>();
            bool Dropout;
            int Depth;
            /// <summary>
            /// 
            /// </summary>
            /// <param name="hiddenSize">Hiddden Dimention</param>
            /// <param name="inSize">Input Dimention</param>
            /// <param name="depth">GRU Layers Depth</param>
            /// <param name="dropout">Use DropOut</param>
            public GRU(int hiddenSize ,int? inSize = null, int depth=1,bool dropout=false)
            {
                this.Depth = depth;
                Dropout = dropout;
                var cell = new GRUCell(hiddenSize, inSize);
                gru.Add(cell);
                Layers.Add(cell);
                for (int i = 1; i < depth; i++)
                {
                    cell = new GRUCell(hiddenSize, hiddenSize);
                    gru.Add(cell);
                    Layers.Add(cell);
                }
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="inputs">Sequenyial Vector Data</param>
            /// <param name="Train">Use in Training</param>
            /// <returns></returns>
            public (Variable　h,List<Variable> outputs) Forward(List<Variable>  inputs,bool Train=false)
            {

                int SeqLen = inputs.Count();
                List<Variable> outputs=new List<Variable>();
                for(int i=0;i<SeqLen;i++)
                {
                    Variable output= inputs[i];
                    for (int j = 0; j < this.Depth; j++)
                    {
                        output = gru[j].Forward(output);
                    }
                    if (Dropout&Train)
                        output = F.DropOut(output,true,0.2);
                    outputs.Add(output);
                }
                return (outputs[SeqLen-1], outputs);
            }
            public (Variable h, List<Variable> outputs) Forward(List<Variable> inputs,Variable oldH, bool Train = false)
            {

                int SeqLen = inputs.Count();
                List<Variable> outputs = new List<Variable>();
                //1周目の処理
                Variable output = inputs[0];
                for (int j = 0; j < this.Depth; j++)
                {
                    output = gru[j].Forward(output,oldH);
                }
                if (Dropout & Train)
                    output = F.DropOut(output, true, 0.2);
                outputs.Add(output);
                //2周目以降の処理
                for (int i = 1; i < SeqLen; i++)
                {
                     output = inputs[i];
                    for (int j = 0; j < this.Depth; j++)
                    {
                        output = gru[j].Forward(output);
                    }
                    if (Dropout & Train)
                        output = F.DropOut(output, true, 0.2);
                    outputs.Add(output);
                }
                return (outputs[SeqLen - 1], outputs);
            }
            public Variable  Forward(Variable input,Variable oldH,bool Train=false)
            {
               
                List<Variable> outputs = new List<Variable>();
                Variable output = input;
                for (int j = 0; j < this.Depth; j++)
                {
                    output = gru[j].Forward(output,oldH);
                }
                if (Dropout&Train)
                    output = F.DropOut(output, true, 0.2);
                outputs.Add(output);
                return output;
            }
            

            /// <summary>
            /// Reset h
            /// </summary>
            public void ResetStatu()
            {
                for (int i = 0; i < Depth; i++)
                {
                    gru[i].h = null;
                }
            }
            public override IEnumerable<Parameter> GetParam()
            {
                foreach (var layer in gru)
                {
                    foreach (var param in layer.GetParam())
                        yield return param;
                }
                foreach(var param in base.GetParam())
                yield return param;
            }

        }
   
        public class ContextGRU : Layer
        {

            List<ContextGRUCell> gru = new List<ContextGRUCell>();
            bool Dropout;
            int Depth;
            /// <summary>
            /// 
            /// </summary>
            /// <param name="hiddenSize">Hiddden Dimention</param>
            /// <param name="inSize">Input Dimention</param>
            /// <param name="depth">GRU Layers Depth</param>
            /// <param name="dropout">Use DropOut</param>
            public ContextGRU(int hiddenSize,int ContextSize, int? inSize = null, int depth = 1, bool dropout = false)
            {
                this.Depth = depth;
                Dropout = dropout;
                var cell = new ContextGRUCell(hiddenSize,ContextSize, inSize);
                gru.Add(cell);
                Layers.Add(cell);
                for (int i = 1; i < depth; i++)
                {
                    cell = new ContextGRUCell(hiddenSize,ContextSize, hiddenSize);
                    gru.Add(cell);
                    Layers.Add(cell);
                }
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="inputs">Sequenyial Vector Data</param>
            /// <param name="Train">Use in Training</param>
            /// <returns></returns>
            public (Variable h, List<Variable> outputs) Forward(List<Variable> inputs, Variable Context,bool Train = false)
            {

                int SeqLen = inputs.Count();
                List<Variable> outputs = new List<Variable>();
                for (int i = 0; i < SeqLen; i++)
                {
                    Variable output = inputs[i];
                    for (int j = 0; j < this.Depth; j++)
                    {
                        output = gru[j].Forward(output);
                    }
                    if (Dropout & Train)
                        output = F.DropOut(output, true, 0.2);
                    outputs.Add(output);
                }
                return (outputs[SeqLen - 1], outputs);
            }
            /// <summary>
            /// 
            /// </summary>
            /// <param name="input">input</param>
            /// <param name="Context">EncorderOutput</param>
            /// <param name="Train">use in Train</param>
            /// <returns></returns>
            public Variable Forward(Variable input,  Variable Context,bool Train = false)
            {

                List<Variable> outputs = new List<Variable>();
                Variable output = input;
                for (int j = 0; j < this.Depth; j++)
                {
                    output = gru[j].Forward(output);
                }
                if (Dropout & Train)
                    output = F.DropOut(output, true, 0.2);
                outputs.Add(output);
                return output;
            }


            /// <summary>
            /// Reset h
            /// </summary>
            public void ResetStatu()
            {
                for (int i = 0; i < Depth; i++)
                {
                    gru[i].h = null;
                }
            }
            public override IEnumerable<Parameter> GetParam()
            {
                foreach (var layer in gru)
                {
                    foreach (var param in layer.GetParam())
                        yield return param;
                }
                foreach (var param in base.GetParam())
                    yield return param;
            }

        }

        /*AttentionUnit*/
        [Serializable]
        public class SelfAttention : Layer
        {
            Linear In;
            Linear Mem1;
            Linear Mem2;
            Linear Out;
            int HDim;

            public SelfAttention(int hdim)
            {
                this.HDim = hdim;
                In= new Linear(hdim);
                Mem1 = new Linear(hdim);
                Mem2 = new Linear(hdim);
                Out = new Linear(hdim);
                Layers.Add(In);
                Layers.Add(Mem1);
                Layers.Add(Mem2);
                Layers.Add(Out);
        }
            public Variable Forward(Variable input,Variable memory)
            {
                var value = Mem2.Forward(memory);
                var query = In.Forward(input) / Math.Sqrt(HDim);
                var logit = Dot(query, Mem1.Forward(memory).Transpose());
                var attentionWeight = SoftMax(logit);
                var output = Dot(attentionWeight, value);
                return Out.Forward(output)+input;
               

            }
        }

        /*CNN Function */
        /// <summary>
        /// 実装予定
        /// </summary>
        public class Conv2d : Layer
        {
            public override Variable4D Forward(Variable4D x)
            {
                return x;
            }
            public Matrix im2Col(Matrix[,] mat, int filter, int stride, int pad)
            {
                int N = mat.GetLength(0);
                int C = mat.GetLength(1);
                int H = mat[0, 0].Row;
                int W = mat[0, 0].Column;

                int outH = (H + 2 * pad - filter) / stride + 1;
                int outW = (W + 2 * pad - filter) / stride + 1;
                var img = new Matrix[N, C];
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < C; j++)
                    {
                        img[i, j] = mat[i, j].MatPad(pad, pad);
                    }
                }
                var col = CreateMatrix(N * outH * outW, C * filter * filter);
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < C; j++)
                    {
                        int down = (img[0, 0].Row - filter) / stride;
                        int right = (img[0, 0].Column - filter) / stride;
                        for (int k = 0; k <= down; k++)
                        {
                            for (int l = 0; l <= right; l++)
                            {
                                //1チャネル内のデータ探索
                                for (int a = 0; a < filter; a++)
                                {
                                    for (int b = 0; b < filter; b++)
                                        col.Data[i * outH * outW + k * outW + l, j * filter * filter + a * filter + b] = img[i, j].Data[k * stride + a, l * stride + b];
                                }
                            }
                        }
                    }
                }
                return col;

            }
            public void Backward()
            { }
        }



    }

   
        

}
