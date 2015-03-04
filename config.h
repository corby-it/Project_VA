#pragma once

const bool processALL = false;
const int waitTimeSpan = 1;
const double learningRate = 0.06;
const bool test = true; //da settare: TRUE se si vuole testare, FALSE se si vogliono creare i file di train

const int lk_thresh = 150; //livello di sicurezza minimo per dare in output la classificazione
const int windowSize = 30;
const int windowNum = 4;
const int windowsStep = 5;