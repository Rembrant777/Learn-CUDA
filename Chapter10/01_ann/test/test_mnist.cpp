#include "../src/mnist.h"
#include "../src/network.h"
#include <gtest/gtest.h>

using namespace std; 
using namespace cudl; 

void print_blob_metadata(Blob<float> *blob) {
    if (blob == nullptr) {
        std::cout<< "NullPtr Blob!" << std::endl; 
        return; 
    }

    // print n, c, h, w 
    std::cout << "Blob#n " << blob->n() << std::endl; 
    std::cout << "Blob#c " << blob->c() << std::endl; 
    std::cout << "Blob#h " << blob->h() << std::endl; 
    std::cout << "Blob#w " << blob->w() << std::endl; 
}

TEST(TestMnist, TestCreateAndInitMnist) {
    MNIST* mnist = new MNIST("./dataset"); 
    EXPECT_NE(mnist, nullptr); 

    // use train's parameters to test 
    int batch_size_train = 256; 
    int num_steps_train = 1600; 

    int monitoring_step = 200;
    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    // use MNIST instance invoke train step 
    mnist->train(batch_size_train, true);

    Blob<float>* train_data = mnist->get_data();
    Blob<float>* train_target = mnist->get_target(); 
    
    EXPECT_EQ(train_data->n(), 256);
    EXPECT_EQ(train_data->c(), 1);
    EXPECT_EQ(train_data->h(), 28);
    EXPECT_EQ(train_data->w(), 28);

    EXPECT_EQ(train_target->n(), 256);
    EXPECT_EQ(train_target->c(), 10);
    EXPECT_EQ(train_target->h(), 1);
    EXPECT_EQ(train_target->w(), 1);

    print_blob_metadata(train_data);
    print_blob_metadata(train_target); 

    delete mnist; 
}


// in this test we execute the network's training operation
TEST(TestMnist, TestMnistTrain) {
    MNIST* mnist = new MNIST("./dataset"); 
    EXPECT_NE(mnist, nullptr); 

    Network model;
    model.add_layer(new Dense("dense1", 500));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense("dense2", 10));
    model.add_layer(new Softmax("softmax"));
    model.cuda();

    // use train's parameters to test 
    int batch_size_train = 256; 
    int num_steps_train = 1600; 

    int monitoring_step = 200;
    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    // use MNIST instance invoke train step 
    mnist->train(batch_size_train, true);

    Blob<float>* train_data = mnist->get_data();
    Blob<float>* train_target = mnist->get_target(); 
    
    EXPECT_EQ(train_data->n(), 256);
    EXPECT_EQ(train_data->c(), 1);
    EXPECT_EQ(train_data->h(), 28);
    EXPECT_EQ(train_data->w(), 28);

    EXPECT_EQ(train_target->n(), 256);
    EXPECT_EQ(train_target->c(), 10);
    EXPECT_EQ(train_target->h(), 1);
    EXPECT_EQ(train_target->w(), 1);

    print_blob_metadata(train_data);
    print_blob_metadata(train_target); 

    mnist->get_batch(); 

    int tp_count = 0; 
    int step = 0; 

    // copy data from host to gpu 
    train_data->to(cuda);
    train_target->to(cuda); 
    
    // let network execute forward calculation 
    model.forward(train_data); 

    // retrieve accuracy value 
    int iter_1_accu = model.get_accuracy(train_target); 

    // let network execute backward calculation 
    // after we execute the backward operation the updated weights and bias values will be 
    // stored network(model) each layer's grad_input 
    model.backward(train_target); 

    // then we execute the network's update operation passing the given learning_rate value 
    // then the backward generated grad_input value will be continue calculated with the learning_rate
    // w' = w - (learn_rate) * dL/dw (1)
    // b' = b - (learn_rate) * dL/db (2)
    // in backward dL/dw and dL/db will be calculated, 
    // d is refering to the derivation, L means the loss function, w means the weight, b means bias 

    // in this update network will iterate its inner layers one by one 
    // and execute each layer#update_weights_biases to execute the formular calculation of (1) and (2)
    model.update(learning_rate);

    // write model's inner trained data to local file 
    model.write_file(); 

    delete mnist; 
}

// in this test we execute the network's test operation
TEST(TestMnist, TestMnistTestPeriod) {
    MNIST* mnist = new MNIST("./dataset"); 
    EXPECT_NE(mnist, nullptr); 

    Network model;
    model.add_layer(new Dense("dense1", 500));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense("dense2", 10));
    model.add_layer(new Softmax("softmax"));
    model.cuda();

    // use train's parameters to test 
    int batch_size_train = 256; 
    int num_steps_train = 1600; 

    int monitoring_step = 200;
    double learning_rate = 0.02f;
    double lr_decay = 0.00005f;

    bool load_pretrain = false;
    bool file_save = false;

    int batch_size_test = 10;
    int num_steps_test = 1000;

    // phase 1 training 
    // use MNIST instance invoke train step 
    mnist->train(batch_size_train, true);

    Blob<float>* train_data = mnist->get_data();
    Blob<float>* train_target = mnist->get_target(); 
    
    EXPECT_EQ(train_data->n(), 256);
    EXPECT_EQ(train_data->c(), 1);
    EXPECT_EQ(train_data->h(), 28);
    EXPECT_EQ(train_data->w(), 28);

    EXPECT_EQ(train_target->n(), 256);
    EXPECT_EQ(train_target->c(), 10);
    EXPECT_EQ(train_target->h(), 1);
    EXPECT_EQ(train_target->w(), 1);

    print_blob_metadata(train_data);
    print_blob_metadata(train_target); 

    mnist->get_batch(); 

    int tp_count = 0; 
    int step = 0; 

    // copy data from host to gpu 
    train_data->to(cuda);
    train_target->to(cuda); 
    
    // let network execute forward calculation 
    model.forward(train_data); 

    // retrieve accuracy value 
    int iter_1_accu = model.get_accuracy(train_target); 

    // let network execute backward calculation 
    // after we execute the backward operation the updated weights and bias values will be 
    // stored network(model) each layer's grad_input 
    model.backward(train_target); 

    // then we execute the network's update operation passing the given learning_rate value 
    // then the backward generated grad_input value will be continue calculated with the learning_rate
    // w' = w - (learn_rate) * dL/dw (1)
    // b' = b - (learn_rate) * dL/db (2)
    // in backward dL/dw and dL/db will be calculated, 
    // d is refering to the derivation, L means the loss function, w means the weight, b means bias 

    // in this update network will iterate its inner layers one by one 
    // and execute each layer#update_weights_biases to execute the formular calculation of (1) and (2)
    model.update(learning_rate);

    // write model's inner trained data to local file 
    model.write_file(); 

    // -- phase 2 inferencing 
    // 1. load test dataset 
    MNIST mnist_test = MNIST("./dataset"); 
    mnist_test.test(batch_size_test);

    // 2. set network to test status 
    model.test(); 

    // 3. create test data and test target from the dataset loader the mnist_test 
    Blob<float>* test_data = mnist_test.get_data(); 
    Blob<float>* test_target = mnist_test.get_target(); 

    mnist_test.get_batch(); 

    // begin testing forward calculating, but first we need to copy data 
    // from host to gpu 
    test_data->to(cuda);
    test_target->to(cuda);

    // execute forward upon the test data 
    model.forward(test_data);
    int test_accuracy = model.get_accuracy(test_target); 
    EXPECT_TRUE(test_accuracy > 0); 
    std::cout << "Test Accuracy Value " << test_accuracy << std::endl; 
}