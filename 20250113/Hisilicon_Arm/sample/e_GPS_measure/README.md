# NmeaParser

此项目演示如何解析 NMEA 协议。

## 如何使用

1. 在源文件中包含“nmeaparser.h”。
2. 创建一个 nmeaparser：'struct nmea_parser parser[1];'。
3. 用“nmea_parser_init（解析器）”初始化“解析器”。
4. 创建导航报表功能：“void my_reporter（结构nav_data *navdata）{...}`.
5. 设置“解析器”的报告功能：“解析器->report_nav_status = my_reporter;”。
6. 从串行端口或文件读取数据时，使用“nmea_parser_putchar（解析器，c）”逐个字符地将这些数据添加到解析器中。
7. 当导航数据准备就绪时，“解析器”将触发“my_reporter”功能来报告导航状态。
## Modules

|模块|文件|描述                         |
| ---------------- | ----------------------------- | ------------------------------------------- |
| `nmea_reader`    | `nmeardr.h` `nmeardr.c`       | 读取 NMEA 数据，包含一个 NMEA 句子。 |
| `nmea_tokenizer` | `nmeatknzr.h` `nmeatknzr.c`   | 将NMEA句子拆分为“令牌”。       |
| `parser`         | `nmeaparser.h` `nmeaparser.c` | 从“令牌”中解析nmea。                 |
| `nav_data`       | `navdata.h` `navdata.c`       | 导航数据。                           |

1. 按字符将 NMEA 数据添加到“nmea_reader”。
2. 如果遇到“\n”，请使用“nmea_tokenizer”拆分“nmea_reader”。
3. 解析“nmea_tokenizer”，并将结果存储在“nav_data”。中。
4. 如果遇到“GGA”句子，请打印“nav_data”。
## 导航数据

#### 日期和时间
日期和时间以 UTC 为单位。

#### 纬度

以度为单位。

北方为正，南方为负。

#### 经度

以度为单位。

对东方为正，对西方为负。

#### Satellite 的 PRN 和 SVID

“PRN”是卫星的NO。在NMEA。

“SVID”是卫星数组中的索引。

|SVID 范围 |星座类型 |
| ---------- | ------------------ |
| 1-64       | GPS                |
| 65-96      | GLONASS            |
| 201-264    | Beidou             |



