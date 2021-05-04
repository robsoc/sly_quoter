# Sly Quoter
> The market-making algorithm built on top of Order Book Pressure features. 

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Coding style](#coding-style)
* [Status](#status)
* [Inspiration](#inspiration)
* [Contact](#contact)

## General info
The purpose of this project is to share the toy market-making model that produces the bid/ask quotes given several user defined inputs 
and some arbitrary assumptions. The model is not optimized to be used for live trading, but rather served as a base idea during strategy
research phase.

The model usage has been presented in the Crypto Currency market, and more precisely on IOTA/BTC pair.

## Technologies
* Python - version 3.6

## Setup
The main class containing whole model logic is stored in `sly_quoter_class.py` script. On the other hand, its usage, particular chunks of 
logic and the rationale of utilized modelling techniques are summarized in the jupyter notebook report `sly_quoter_report.py`. 

In order to install the libraries with appropriate versioning type:

```
pip install requirements.txt
```

## Coding style 
The project follows PEP8 style.

## Status
Project is: _done_.

## Inspiration
Project is inspired by the market making algo research I've done for one of my projects within the algo trading business.

## Contact
Created by [@robsoc](https://github.com/robsoc) - feel free to contact me!
