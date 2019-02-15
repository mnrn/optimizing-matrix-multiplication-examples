#################################################################################
# @brief makefileのテンプレートです...
# @note  GNU Make 3.81で動作確認しました
# @note  以下のサイトを参考にしました
#        http://urin.github.io/posts/2013/simple-makefile-for-clang/
# @note  わからないコマンドがあったらGNU Make(O'reilly)を参考にしてください
# @date  作成日     : 2016/02/16
# @date  最終更新日 : 2016/02/16
#################################################################################



CC      = g++

CFLAGS  = -march=native -mtune=native -fopenmp -std=c++17 -O3 -MMD -MP
WARNS   = -Wall -Wextra -Wconversion -Wdisabled-optimization -Wfloat-equal -Winline -Wreorder

LDFLAGS = -fopenmp 
LIBS    =
INC     = -I./include
TARGET  = ./bin/$(shell basename $(shell dirname `readlink -f .`))

SRCDIR  = ./src
ifeq "$(strip $(SRCDIR))" ""
	SRCDIR = .
endif
SRCS    = $(wildcard $(SRCDIR)/*.cpp)

OBJDIR  = ./obj
ifeq "$(strip $(OBJDIR))" ""
	OBJDIR = .
endif
OBJS    = $(addprefix $(OBJDIR)/, $(notdir $(SRCS:.cpp=.o)))

DEPENDS = $(OBJS:.o=.d)


$(TARGET): $(OBJS) $(LIBS)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-mkdir -p $(OBJDIR)
	$(CC) -o $@ -c $< $(INC) $(CFLAGS) $(WARNS)

all: clean $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) $(DEPENDS)

-include $(DEPENDS)

