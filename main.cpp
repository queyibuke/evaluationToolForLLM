#include<iostream>
#include<string>
#include<cstdint>
#include<cmath>
#include<iomanip>
#include<vector>
#include<fstream>
#if WIN32
    #define YAML_CPP_STATIC_DEFINE
#endif
#include"yaml-cpp/yaml.h"

std::string fileName = {};
std::vector<std::string> option{"1 estimate memory usage", "2 estimate traintime", "3 training parameter optimization"};
int opt = 0;

int seqLen = 0;//序列长度
std::string modelSize = {};//模型参数
uint64_t modelSizeInt = 0;//模型参数
int attnHeadNum = 0;//注意力头数
float globalBatchSize = 0;//全局批量大小
float miniBatchSize = 0;//微批量大小
int step = 0;//在一次梯度更新中累积的传播步数
int gpuNumber = 0;//GPU数量
int hiddenLayerDimension = 0;//隐藏层维度
int layer = 0;//层数

int optimizerStrategy = 0;//优化器策略 0:None 1:zero1 2:zero2 3:zero3
int dataParaSize = 0;//数据并行大小
int tensorParaSize = 0;//张量并行大小
int pipelineParaSize = 0;//流水线并行大小
int sequenceParaSize = 0;//序列并行大小

std::string dataNum = {};
float dataNumFloat = 0;
float fPointOp = 0.0;
float MFU = 0.0;

uint64_t memUsage = 0;
uint64_t optimizerMem = 0;
uint64_t activationMem = 0;
float trainTime = 0.0;

void inputFunNum() {
    std::cout << "follow these tips to complete the model evaluation" << std::endl;
    for(auto& a : option ) {
        std::cout << a << std::endl;
    }
    std::cout << "please enter the function number: ";
    std::cin >> opt;
    while(opt <= 0 || opt > option.size()) {
        std::cout << "the number is not in the range, the input is wrong, please re-enter: " ;
        std::cin >> opt;
    }
    return;
}

/**
 * @brief 从 YAML 文件中获取参数
 *
 * 从名为 "para.yaml" 的 YAML 文件中读取参数，并设置相应的成员变量。
 * 如果文件无法打开，则抛出运行时错误。
 */
void getYamlPara(std::string &fileName) {
    std::ifstream file(fileName);
    while(!file.is_open()) {
        throw std::runtime_error("Unable to open file!");
        system("pause");
    }
    YAML::Node config = YAML::LoadFile(fileName);

    seqLen = config["seqLen"].as<int>();
    modelSize = config["modelSize"].as<std::string>();
    modelSizeInt = static_cast<uint64_t>(static_cast<int>(std::stof(modelSize.substr(0, modelSize.size() - 1)) * 1000)) * 1000000;
    attnHeadNum = config["attnHeadNum"].as<int>();
    globalBatchSize = config["globalBatchSize"].as<int>();
    step = config["step"].as<int>();
    gpuNumber = config["gpuNumber"].as<int>();
    hiddenLayerDimension = config["hiddenLayerDimension"].as<int>();
    layer = config["layer"].as<int>();
    optimizerStrategy = config["optimizerStrategy"].as<int>();
    dataParaSize = config["dataParaSize"].as<int>();
    tensorParaSize = config["tensorParaSize"].as<int>();
    pipelineParaSize = config["pipelineParaSize"].as<int>();
    sequenceParaSize = config["sequenceParaSize"].as<int>();

    dataNum = config["dataNum"].as<std::string>();
    dataNumFloat = std::stof(dataNum.substr(0, dataNum.size() - 1));
    fPointOp = config["fPointOp"].as<float>();
    MFU = config["MFU"].as<float>();

    std::cout << "the para.yaml para is:" << std::endl;
    std::cout << "seqLen is " << seqLen << std::endl
        << "modelSize is " << modelSizeInt << '(' << modelSize << ')' << std::endl
        << "attnHeadNum is " << attnHeadNum << std::endl
        << "globalBatchSize is " << globalBatchSize << std::endl
        << "step is " << step << std::endl
        << "gpuNumber is " << gpuNumber << std::endl
        << "hiddenLayerDimension is " << hiddenLayerDimension << std::endl
        << "layer is " << layer << std::endl;
    if(optimizerStrategy == 0) {
        std::cout << "optimizerStrategy is None" << std::endl;
    } else {
        std::cout << "optimizerStrategy is Zero" << optimizerStrategy << std::endl;
    }
    std::cout<< "dataParaSize is " << dataParaSize << std::endl
        << "tensorParaSize is " << tensorParaSize << std::endl
        << "pipelineParaSize is " << pipelineParaSize << std::endl
        << "sequenceParaSize is " << sequenceParaSize << std::endl
        << "dataNum is " << dataNumFloat << '(' << dataNum << ')' << std::endl
        << "fPointOp is " << fPointOp << std::endl
        << "MFU is " << MFU << std::endl;
}

/**
 * @brief 计算优化器所需的内存大小
 *
 * 根据优化策略计算参数、梯度和优化器状态占用显存
 *
 * @param f_optimizerStrategy 优化器策略
 * @param f_modelSize 模型大小
 * @param f_gpuNmuber GPU 数量
 *
 * @return 优化器所需的内存大小
 */
uint64_t getOptimizerMem(const int f_optimizerStrategy, const uint64_t f_modelSize, const int f_gpuNumber) {
    uint64_t optimizerMem = 0;
    switch (f_optimizerStrategy) {
    case 0:
        optimizerMem = 16 * f_modelSize;
        break;
    case 1:
        optimizerMem = 4 * f_modelSize * f_gpuNumber + 12 * f_modelSize;
        break;
    case 2:
        optimizerMem = 2 * f_modelSize * f_gpuNumber + 14 * f_modelSize;
        break;
    case 3:
        optimizerMem = 16 * f_modelSize;
        break;
    default:
        optimizerMem = 16 * f_modelSize;
        break;
    }
    return optimizerMem;
}

/**
 * @brief 计算激活内存大小
 *
 * 根据给定的参数计算激活内存的大小。
 *
 * @param f_seqLen 序列长度
 * @param f_attnHeadNum 注意力头数
 * @param f_globalBatchSize 全局批处理大小
 * @param f_minibatchSize 迷你批处理大小
 * @param f_step 一次梯度更新中累积的前向传播和反向传播的步数
 * @param f_gpuNumber GPU数量
 * @param f_hiddenLayerDimension 隐藏层维度
 * @param f_layer 层数
 * @param f_dataParaSize 数据并行大小
 * @param f_tensorParaSize 张量并行大小
 * @param f_pipelineParaSize 流水线并行大小
 * @param f_sequenceParaSize 序列并行大小
 *
 * @return 激活内存大小
 */
uint64_t getActivationMem(const int f_seqLen, const int f_attnHeadNum, const int f_globalBatchSize, const int f_minibatchSize,
    const int f_step, const int f_gpuNumber, const int f_hiddenLayerDimension, const int f_layer,
    const int f_dataParaSize, const int f_tensorParaSize, const int f_pipelineParaSize, const int f_sequenceParaSize) {
    
    uint64_t activationMem = 0;
    uint64_t a1 = 0;
    uint64_t a2 = 0;
    int t_seqLens = f_seqLen;
    int t_globalBatchSize = f_globalBatchSize;
    int t_dataParaSize = f_dataParaSize;
    int t_tensorParaSize = f_tensorParaSize;
    int t_pipelineParaSize = f_pipelineParaSize;
    int t_sequenceParaSize = f_sequenceParaSize;

    if(t_dataParaSize <= 1 ) {
        t_dataParaSize = 1;
    } else{
        t_globalBatchSize = t_globalBatchSize / f_step / f_gpuNumber;
    }
    if(t_tensorParaSize <= 1) {
        t_tensorParaSize = 1;
    }
    if(t_pipelineParaSize <= 1) {
        t_pipelineParaSize = 1;
    } else {
        t_globalBatchSize = t_globalBatchSize / t_pipelineParaSize;
    }
    if(t_sequenceParaSize <= 1) {
        t_sequenceParaSize = 1;
    } else {
        t_seqLens = t_seqLens / t_sequenceParaSize;
    }
    a1 = f_layer * t_globalBatchSize * t_seqLens;
    if (t_tensorParaSize == 1) {
        a2 = 34 * f_hiddenLayerDimension  + 5 * t_seqLens * f_attnHeadNum;
        activationMem = a1 * a2 / t_pipelineParaSize;
    } else {
        a2 = (10 + 24 / t_tensorParaSize) * f_hiddenLayerDimension  + 5 * t_seqLens * f_attnHeadNum / t_tensorParaSize;
        activationMem = a1 * a2 / t_pipelineParaSize;
    }
    return activationMem;
}

float getTrainingTime(const float f_dataNum, const uint64_t f_modelSize, const int f_gpuNumber, const float f_fPointOp, const float f_MFU) {
    
    float trainingDay = 0;
    trainingDay = f_dataNum * 6 * f_modelSize / (f_gpuNumber * f_fPointOp * f_MFU ) / 3600 / 24;
    return trainingDay;
}

int main(char **argv, int argc) {
    if(argc != 2) {
        std::cout << "please input the yaml file name: ";
        std::cin >> fileName;
    } else {
        fileName = argv[1];
    }
    getYamlPara(fileName);
    inputFunNum();
    while(1) {
        switch (opt) {
        case 1:
            optimizerMem = getOptimizerMem(optimizerStrategy, modelSizeInt, gpuNumber);
            activationMem = getActivationMem(seqLen, attnHeadNum, globalBatchSize, miniBatchSize, step, gpuNumber, hiddenLayerDimension, layer, dataParaSize, tensorParaSize, pipelineParaSize, sequenceParaSize);
            memUsage = optimizerMem + activationMem;
            std::cout << "memUsage is " << memUsage << "B(" << memUsage / 1000000000 << "GB) " << std::endl;
            break;
        case 2:
            trainTime = getTrainingTime(dataNumFloat, modelSizeInt, gpuNumber, fPointOp, MFU);
            std::cout << "trainTime is " << std::fixed << std::setprecision(1) << trainTime  << std::endl;
            break;
        case 3:
            /* code */
            break;
        default:
            break;
        }
        std::cout << "if you want to exit, please input 'q'. or you want to continue, please input other: ";
        char c = getchar();
        if(c = getchar() == 'q') {
            break;
        } else {
            inputFunNum();
        }       
    }
    system("pause");
    return 0;    
}