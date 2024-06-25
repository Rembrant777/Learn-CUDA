#include "../src/loss.h"
#include "../src/layer.h"
#include "../src/helper.h"
#include <gtest/gtest.h>

using namespace cudl; 
using namespace std;

TEST(TestLayer, DenseLayerCreateTest) {
    string name = "test_dense_layer_1"; 
    int output_size = 10; 
    Dense* layer = new Dense(name, output_size); 
    EXPECT_NE(layer, nullptr); 
    EXPECT_NE(layer, nullptr); 

    delete layer; 
}

TEST(TestLayer, ActivationLayerCreateTest) {
    string name = "test_activation_layer_1"; 
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU; 
    Activation* layer = new Activation(name, mode); 
    EXPECT_NE(layer, nullptr); 

    delete layer; 
}

TEST(TestLayer, SoftmaxLayerCreateTest) {
    string name = "softmax-layer-1"; 
    Softmax* layer = new Softmax(name); 
    EXPECT_NE(layer, nullptr); 

    delete layer; 
}

/**
 In this test, we test 
 1. create Dense(Layer) instance with name of layer-0 and set input data to it 
 2. unfreeze its parameter update status(freeze = false means its inner parameter(weight and bias) can be modifed)
 3. invoke fwd_initialize function to invoke its inner init_weights_bias function to generate params to its memory space
 4. invoke its save_parameter function to write params to local disk file 
 5. create another Dense(Layer) instance with the name of layer-1 
 6. unfreeze layer-1 instances status and also enable its load_pretrain variable from false to true which enables 
    load parameters from local files when invoke fwd_initialize 

test results: layer-0's weight and bias value match with the layer-1's weight and bias value     
*/
TEST(Testlayer, DenseInitWeightBias) {
    string d_name = "dense-layer-0"; 
    int output_size = 8; 
    Dense* d_layer_0 = new Dense(d_name, output_size); 
    EXPECT_NE(d_layer_0, nullptr);

    int n = 1, c = 2, h = 3, w = 4; 
    Blob<float>* input = new Blob<float>(n, c, h, w); 
    d_layer_0->unfreeze(); 
    d_layer_0->fwd_initialize(input); 
    

    // init biases cannot be null 
    Blob<float>* p_biases = d_layer_0->biases_; 
    EXPECT_NE(p_biases, nullptr); 

    // init weights cannot be null
    Blob<float>* p_weights = d_layer_0->weights_; 
    EXPECT_NE(p_weights, nullptr); 

    // init input_ cannot be null 
    // input's tensor (input data dimension meta info object) neither cannot be null 
    EXPECT_EQ(input, d_layer_0->input_);
    EXPECT_NE(nullptr, d_layer_0->input_->tensor()); 

    // init output_ cannot be null 
    EXPECT_NE(nullptr, d_layer_0->output_);
    // output's tensor descriptor neither cannot be null
    EXPECT_NE(nullptr, d_layer_0->output_->tensor()); 

    // init forward will create & init d_one_vec 
    EXPECT_NE(nullptr, d_layer_0->d_one_vec); 

    // weight & bias init value are initialized in fwd_initialize 
    // d_layer->init_weight_bias(); 
    p_biases->print("bias-value", 1, 4); 
    p_weights->print("weight-value", 1, 4); 


    // here we check class variable load_pretrain_ 
    // if load_pretrain_ = true it will load parameters(weight & bias) from file 
    // otherwise dense layer will invoke init_weights_biases function to generate parameters and hold them in memory 
    EXPECT_EQ(false, d_layer_0->load_pretrain_); 

    // since every thing is ok, we mock the network layer's write its parameters to disk
    // then let next layer to load parameter from the disk 
    int ret = d_layer_0->save_parameter(); 
    // if parameters are write success, ret value should be 
    EXPECT_EQ(0, ret); 

    // weight parameter file should be: dense-layer-0.bias.bin 
    // bias file should be: dense-layer-0.bias.bin 

    // then to test load_parameter works as expected, 
    // we create a new dense layer instance here and set its freeze_ = false and load_pretrain_ = true 
    // in this way it will load the parameter file that previous layer(point by the d_layer) write 
    string d_name_1 = "dense-layer-0";  
    Dense* d_layer_1 = new Dense(d_name_1, output_size); 
    EXPECT_NE(nullptr, d_layer_1);

    d_layer_1->set_load_pretrain();
    EXPECT_EQ(true, d_layer_1->load_pretrain_); 

    d_layer_1->unfreeze();
    // unfreeze here means weight and bias parameters can be modified 
    // freeze means parameters cannot be modified 
    EXPECT_EQ(false, d_layer_1->freeze_);

    Blob<float>* layer_1_input = new Blob<float>(n, c, h, w); 
    EXPECT_NE(nullptr, layer_1_input);
    d_layer_1->fwd_initialize(layer_1_input);

    EXPECT_NE(d_layer_1->weights_, nullptr); 
    EXPECT_NE(d_layer_1->biases_, nullptr);

    // print disk file loaded parameters 
    d_layer_1->weights_->print("layer1-weights", 1, w); 
    d_layer_1->biases_->print("layer1-biases", 1, w); 

    // layer0 and layer1 weight param match 
    EXPECT_EQ(true, d_layer_1->weights_->equals(d_layer_0->weights_)); 
    EXPECT_EQ(true, d_layer_1->weights_->equals(d_layer_1->weights_)); 

    // self should be equal 
    EXPECT_EQ(true, d_layer_1->biases_->equals(d_layer_1->biases_)); 

    // layer0 and layer1 bias param match 
    EXPECT_EQ(true, d_layer_1->biases_->equals(d_layer_0->biases_)); 

    delete d_layer_0; 
    delete d_layer_1; 
}

/**
 In this ut we create instance of Dense and generate mock data for input data
 let Dense invoke forward function to calculate the output = weights^T * input
*/ 
TEST(Testlayer, DenseLayerForwardExecute) {
    int n = 1, c = 2, h = 3, w = 4; 
    int output_size = w; 

    Blob<float>* input = new Blob<float>(n, c, h, w); 
    EXPECT_NE(input, nullptr); 

    // create cuda context 
    CudaContext* cuda_context = new CudaContext(); 
    EXPECT_NE(cuda_context, nullptr); 

    // invoke method to generate mock input dataset 
    input->gen_mock_data_for_predict(); 

    // print generated test dataset 
    // cannot invoke print() here because print() function 
    // will invoke cuda mem -> overwrite host mem -> all generated mock data will be overwrited. 
    input->print_data("input_dataset", n, w); 

    string name = "dense-layer-0"; 
    Dense* d_layer = new Dense(name, output_size); 

    // set cuda context to layer instance 
    d_layer->set_cuda_context(cuda_context);

    // init weight and output data 
    d_layer->fwd_initialize(input); 

    EXPECT_NE(d_layer, nullptr); 

    // invoke forward calculation 
    std::cout << "invoke forward func" << std::endl; 
    Blob<float>* output = d_layer->forward(input); 
    EXPECT_NE(output, nullptr); 
    EXPECT_NE(output->ptr(), nullptr); 
    for (int i = 0; i < output->size(); i++) {
        std::cout << output->ptr()[i] << std::endl; 
    }

    // print data of output 
    // output->print_data("output_value");

    delete d_layer; 
}

/**
 In this ut we init backward calculation context and apply data space
*/
TEST(Testlayer, DenseLayerBackwardExecute)  {
    int n = 1, c = 2, h = 3, w = 4; 
    int output_size = w; 

    Blob<float>* grad_output = new Blob<float>(n, c, h, w);
    EXPECT_NE(grad_output, nullptr); 
    grad_output->gen_mock_data_for_predict(); 
    grad_output->print_data("grad_output", n, w); 
    
    Blob<float>* input = new Blob<float>(n, c, h, w); 
    EXPECT_NE(input, nullptr); 
    input->gen_mock_data_for_predict(); 
    input->print_data("input", n, w); 

     // create cuda context 
    CudaContext* cuda_context = new CudaContext(); 
    EXPECT_NE(cuda_context, nullptr);

    string name = "dense-layer"; 
    Dense* d_layer = new Dense(name, output_size); 

    d_layer->set_cuda_context(cuda_context); 

    // init backward data 
    d_layer->fwd_initialize(input); 
    EXPECT_NE(d_layer->weights_, nullptr); 
    EXPECT_NE(d_layer->biases_, nullptr);

    d_layer->bwd_initialize(grad_output); 
    EXPECT_NE(d_layer->grad_weights_, nullptr); 
    EXPECT_NE(d_layer->grad_biases_, nullptr); 
    EXPECT_NE(d_layer->grad_input_, nullptr); 
    EXPECT_NE(d_layer->grad_output_, nullptr); 

    Blob<float>* grad_input = d_layer->backward(grad_output); 
    EXPECT_NE(grad_input, nullptr); 

    std::cout << "Print Gradient Input " << std::endl; 
    grad_input->print_data("gradient input data", 1, w); 

    delete d_layer; 
}

/**
 In this test case we execute 
 1. activaiton layer forward init operaiton unit tests
 2. activation layer forward calculation  
*/
TEST(Testlayer, ActivationLayerFwdInit) {
    int n = 1, c = 2, h = 3, w = 5; 
    Blob<float>* input = new Blob<float>(n, c, h, w);
    EXPECT_NE(input, nullptr); 
    
    string name = "activation-layer"; 
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID; 
    float coef = 0.0; 

    Activation* layer = new Activation(name, mode, coef); 
    EXPECT_NE(layer, nullptr);

    // after create activation's inner activation descriptor cannot be null 
    cudnnActivationDescriptor_t act_desc = layer->get_act_desc(); 
    EXPECT_NE(&act_desc, nullptr); 

    // here begin init forward, input structure applied is ok 
    layer->fwd_initialize(input); 

    // check fwd_initialize inner value are initialized 
    EXPECT_NE(layer->input_desc_, nullptr); 

    EXPECT_NE(layer->input_, nullptr);

    EXPECT_NE(layer->output_desc_, nullptr); 
    EXPECT_NE(layer->output_, nullptr); 

    // then we invoke Blob's inner function to generate mock input data
    input->gen_mock_data_for_predict(); 
    input->print_data("activation_fwd_input_data", 1, input->w()); 

    // create cuda context 
    CudaContext* cuda_context = new CudaContext(); 
    layer->set_cuda_context(cuda_context); 
    
    Blob<float>* output = layer->forward(input); 
    EXPECT_NE(output, nullptr); 
    output->print_data("activation_fwd_output_data", 1, output->w()); 
    delete layer; 
}

/**
 In this test case, we execute 
 1. activation backward create & init operation
 2. activation backward calculate operation 
*/
TEST(Testlayer, ActivationLayerBwdExecute) {
     int n = 1, c = 2, h = 8, w = 9; 
     // create input && grad_output Blob instance 
     Blob<float>* input = new Blob<float>(n, c, h, w); 
     Blob<float>* grad_output = new Blob<float>(n, c, h, w); 

     EXPECT_NE(input, nullptr); 
     EXPECT_NE(grad_output, nullptr);

     string name = "activation layer"; 
     // select sigmode func as the activation function 
     cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU; 
    
     Activation* layer = new Activation(name, mode);  
     EXPECT_NE(layer, nullptr);

     // init the Activation Layer's fwd and bwd env 
     layer->fwd_initialize(input); 

     // verify fwd initialized member variables 
     EXPECT_NE(layer->input_, nullptr); 
     EXPECT_NE(layer->output_, nullptr);
     EXPECT_NE(layer->input_desc_, nullptr);  
     EXPECT_NE(layer->output_desc_, nullptr); 

     layer->bwd_initialize(grad_output); 
     EXPECT_NE(layer->grad_input_, nullptr);
     EXPECT_NE(layer->grad_output_, nullptr); 

     // verify bwd initialized member variables 
     // all bwd required member variables are initialized during fwd init period 

     // create cuda context 
     CudaContext* cuda_context = new CudaContext(); 
     EXPECT_NE(cuda_context, nullptr); 
     layer->set_cuda_context(cuda_context); 

     // let grad_output and input generate mock dataset 
     input->gen_mock_data_for_predict(); 
     grad_output->gen_mock_data_for_predict(); 

     Blob<float>* output = layer->forward(input); 
     EXPECT_EQ(output, nullptr); 
     output->print_data("forward_output_data", n, w); 

     // invoke bwd calculation 
     Blob<float>* grad_input = layer->backward(grad_output); 
     EXPECT_EQ(grad_input, nullptr);
     grad_input->print_data("backward_grad_input_data", n, w); 

    delete layer; 
}