// Initial commit; once code is done in Python will transition to C++ for performance


#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
// #include <string>
// #include <unistd.h>
// #include <iostream>
// #include <vector>

using namespace std;

int main() {
	
	ifstream ip;
	ip.open("...");

	if(!ip.is_open()) std::cout << "ERROR: File Open" << "\n";

	string ticker, date;
	// double open, high, low, close, adj_close, volume;
	// string open, high, low, close, adj_close, volume;
	// int i = 0;
	// float open, high, low, close, adj_close, volume;

	// double funds = 100000;

	// std::cout << "What Ticker?\n";
	
  	// std::cin >> ticker;
	// while(ip.good()) {
	// 	getline(ip,ticker,',');
	// 	getline(ip,date,',');
	// 	getline(ip,open,',');
	// 	getline(ip,high,',');
	// 	getline(ip,low,',');
	// 	getline(ip,close,',');
	// 	getline(ip,adj_close,',');
	// 	getline(ip,volume,'\n');

	// if (ticker == "AAPL") {
	// 	for (i = 0; i < 1; i++) {
		
	// 	// while(ticker == "AAPL" && date == "1990-01-22") {
	// 		std::cout << i << "Ticker: " << ticker << " \n";
	// 		std::cout << i << "Date: " << date << " \n";
	// 		std::cout << i << "Open: " << open << " \n";
	// 		std::cout << i << "Close: " << close << " \n";
	// 		}
	// 	}
	// // if (ticker != "AAPL") {
	// // 	break;
	// // }
	// }
	ip.close();
}
