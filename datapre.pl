#4个模型的数据处理，包括真数据和模拟假数据的生成。

#!/usr/bin/perl
$annotation_path = '.././data/homo_sapiens/trans';     #读入gtf注释文件
$hg19_path = '.././data/hg19.fa';                      
$bp = 64;                                              #每个剪接位点用以剪接位点为中心的64bp碱基序列表示
$ban = 33;                                             #$bp/2+1
#open source data
open gene,"<$hg19_path" or die$!;
open annotation,"<$annotation_path" or die$!;

open data1,">.././data64/splite_site_64bp" or die$!;
open data2,">.././data64/exon_ends_64bp" or die$!;
open data3,">.././data64/exon_unit_64bp" or die$!;
open data4,">.././data64/transcript" or die$!;
print "###### hg19 hash making####\n";
print data4 "gene\n";
$chr='chr1';
while(<gene>)
{      chomp;
      if ($_ =~ /^>(.*)/)
          {
                $gene{$chr}=$seq;

        $chr = $1;
                if($1 eq 'chrM')
                {
                        last;
                }
        $seq = "";
      }else{
                $seq .= $_;
      }
}
$gene{$chr} = $seq;
print "####### hg19 hash made####\n";
#循环读取
$number=0;
while($line = <annotation>)
{
    G:$exon_i=0;
        $exon_j=0;
        %exon_unit=();
        %exon_site=();

        chomp $line;
        @words=split(" ",$line);
        if($words[2] ne 'gene')
        {
                next;
        }
        else 
        {
                if($words[6] eq '+')
                {
                        $mode = 0;
                }
                else 
                {
                        $mode = -1;
                }
                #修改染色体
                $chr = 'chr'.$words[0];
                unless(exists $gene{$chr})
                {
                        last;
                }
                #记录gene起始终止位置
                $i = ($words[3],$words[4])[$mode+0];
                $j = ($words[3],$words[4])[$mode+1];

                #读入下一行，，应该是transcript
                while($tran = <annotation>)
                {
                        #这个地方要初始化很多参数
                        T:$exon_i=0;
                        $exon_j=0;
                        chomp $tran;
                        @tran_words = split(" ",$tran);
                        if($tran_words[2] eq 'transcript')
                        {
                                if(++$number%10000 == 0)
                                {
                                        print "#####已经处理了 $number 个transcript\n";
                                }
                        
                                $tran_i = ($tran_words[3],$tran_words[4])[0+$mode];
                                $tran_j = ($tran_words[4],$tran_words[4])[1+$mode];
                                #读取下一行应该是exon，如果是transcript 或者gene 跳转
                                while($exon = <annotation>)
                                {
                                        chomp $exon;
                                        @exon_words = split(" ",$exon);
                                        if($mode == -1)
                                        {
                                                ($exon_words[3],$exon_words[4]) = ($exon_words[4],$exon_words[3]);
                                        }
                                        if ($exon_words[2] eq 'exon')
                                        {
                                                #我想把他们都存到一个hash里面 exon_site hash 存剪接位点 exon_ends 存一个exon的两端
                                                if ($exon_j)
                                                {
                                                        push (@exon_unit,($exon_j,$exon_words[3]));#@exon_unit is a array which the odd number is the last exon ends site
                                                        $exon_unit{$exon_j.'_'.$exon_words[3]}++;

                                                }
                                                else
                                                {
                                                        #如果是第一个exon 则 定一个起点 @
                                                        push (@exon_unit,('@',$exon_words[3]));
                                                        $exon_unit{'@'.'_'.$exon_words[3]}++;

                                                }
                                                $exon_i = $exon_words[3];
                                                $exon_j = $exon_words[4];
                                                $exon_site{$exon_i."_".$exon_j}++;
                                        }
                                        else 
                                        {
                                                #这个地方用了goto 不知道用的对不对,maybe right by using the templete code b.pl
                                                if (($exon_words[2] eq 'gene') or ($exon_words[2] eq 'transcript'))
                                                {
                                                        #这个exon_unit结束
                                                        push (@exon_unit,($exon_j,'#'));
                                                        #记录剪切图
                                                        push @transcriptgraph,[@exon_unit];
                                                        $transtraph{join("_",@exon_unit)}++;

                                                        $exon_unit{$exon_j.'_'.'#'}++;
                                                        #输出文件 like @        1       4       7        #      1
                                                        #                       @       1       7       #       1
                                                        
                                                        print data4 "@exon_unit\t";
                                                        print data4 "1\t$chr\n";
                                                        #删除这个数组
                                                        splice(@exon_unit,0);
                                                        if ($exon_words[2] eq 'gene')
                                                        {
                                                                
                                                                print data4 "gene\n";
                                                                ################################################
                                                                foreach $ei (keys %exon_site)
                                                                {
                                                                        #打印exon
                                                                        @ej = split("_",$ei);

                                                                        #print data2 substr($gene{$chr},$ej-$ban,$bp)."\t";

                                                                        $seq1 = substr($gene{$chr},$ej[0]-$ban,$bp);
                                                                        $seq2 = substr($gene{$chr},$ej[1]-$ban,$bp);
                                                                        if($mode == -1)
                                                                        {
                                                                                $seq1 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq1 = reverse $seq1;
                                                                                $seq2 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq2 = reverse $seq2;
                                                                        }
                                                                        print data2 "$seq1\t$seq2\t$ej[0]\t$ej[1]\t";
                                                                        $long = abs($ej[1]-$ej[0]);
                                                                        print data2 "1\t$long\n";
                                                                        #for fake exon_ends
                                                                        
                                                                        push(@pre_exon,$ej[0]);
                                                                        push(@beh_exon,$ej[1]);
                                                                        
                                                                        #打印split site
                                                                        #print data1 substr($gene{$chr},$ej[0]-$ban,$bp)."\t0\n";
                                                                        #print data1 substr($gene{$chr},$ej[1]-$ban,$bp)."\t1\n";
                                                                        print data1 $seq1."\t$ej[0]\t0\n";
                                                                        print data1 $seq2."\t$ej[1]\t1\n";

                                                                        #通过随机数可以产生不同数量的fake数据
                                                                        if(rand() > 0.5)
                                                                        {
                                                                                $site=int(rand(abs($ej[1]-$ej[0])-1))+1+$ej[0+$mode];
                                                                                #print data1 substr($gene{$chr},$site-$ban,$bp)."\t2\n";#2代表exon之间的
                                                                                $seq = substr($gene{$chr},$site-$ban,$bp);
                                                                                if($mode == -1)
                                                                                {
                                                                                        $seq =~tr/atcgATCG/tagcTAGC/;
                                                                                        $seq = reverse $seq;
                                                                                        
                                                                                }
                                                                                print data1 $seq."\t$site\t2\n";
                                                                        }
                                                                                
                                                                }
                                                                
                                                                $num=@pre_exon;

                                                                for( $nn = 0; $nn<$num; $nn = $nn + 1 )
                                                                {
                                                                        for($mm = 0 ; $mm < $num ; $mm = $mm + 1)
                                                                        {
                                                                                $intant = $pre_exon[$nn].'_'.$beh_exon[$mm];
                                                                                if(exists $exon_site{$intant})
                                                                                {
                                                                                        next;
                                                                                }
                                                                                else
                                                                                {
                                                                                        $fake_exon_site{$intant}++;
                                                                                        
                                                                                }
                                                                        }
                                                                }
                                                                $fake_i = 1;
                                                                foreach $fake (keys %fake_exon_site)
                                                                {
                                                                        @fakearray = split("_",$fake);
                                                                        #print data2 substr($gene{$chr},$fakearray[0]-$ban,$bp)."\t".substr($gene{$chr},$fakearray[1]-$ban,$bp)."\t";
                                                                        $seq1 = substr($gene{$chr},$fakearray[0]-$ban,$bp);
                                                                        $seq2 = substr($gene{$chr},$fakearray[1]-$ban,$bp);
                                                                        if($mode == -1)
                                                                        {
                                                                                $seq1 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq1 = reverse $seq1;
                                                                                $seq2 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq2 = reverse $seq2;
                                                                        }
                                                                        print data2 "$seq1\t$seq2\t$fakearray[o]\t$fakearray[1]\t";

                                                                        $long = abs($fakearray[1]-$fakearray[0]);
                                                                        print data2 "0\t$long\n";
                                                                        $fake_i = $fake_i+1;
                                                                        if($fake_i > $num/2)
                                                                        {
                                                                                last;
                                                                        }
                                                                }

                                                                #清空用到的变量
                                                                @pre_exon = ();
                                                                @beh_exon = ();
                                                                %fake_exon_site = ();


                                                                #输出exon 之间的连接
                                                                foreach $ei (keys %exon_unit)
                                                                {
                                                                        @ej = split("_",$ei);
                                                                        $seq1 = substr($gene{$chr},$ej[0]-$ban,$bp);
                                                                        $seq2 = substr($gene{$chr},$ej[1]-$ban,$bp);
                                                                        if($mode == -1)
                                                                        {
                                                                                $seq1 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq1 = reverse $seq1;
                                                                                $seq2 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq2 = reverse $seq2;
                                                                        }
                                                                        print data3 "$seq1\t$seq2\t$ej[0]\t$ej[1]\t";
                                                                        #print data3 substr($gene{$chr},$ej-$ban,$bp)."\t";
                                                                        $long = abs($ej[1]-$ej[0]);
                                                                        print data3 "1\t$long\n";
                                                                        
                                                                        push(@pre_exon,$ej[0]);
                                                                        push(@beh_exon,$ej[1]);
                                                                }
                                                                $num=@pre_exon;
                                                                $fakenum = 0;
                                                                for( $nn = 0; $nn<$num; $nn = $nn + 1 )
                                                                {
                                                                        for($mm = 0 ; $mm < $num ; $mm = $mm + 1)
                                                                        {
                                                                                $intant = $pre_exon[$nn].'_'.$beh_exon[$mm];
                                                                                if(exists $exon_unit{$intant})
                                                                                {
                                                                                        next;
                                                                                }
                                                                                else
                                                                                {
                                                                                        $fake_exon_unit{$intant}++;
                                                                                        
                                                                                }
                                                                        }
                                                                }
                                                                $fake_i = 1;
                                                                foreach $fake (keys %fake_exon_unit)
                                                                {
                                                                        @fakej = split("_",$fake);
                                                                        $seq1 = substr($gene{$chr},$fakej[0]-$ban,$bp);
                                                                        $seq2 = substr($gene{$chr},$fakej[1]-$ban,$bp);
                                                                        if($mode == -1)
                                                                        {
                                                                                $seq1 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq1 = reverse $seq1;
                                                                                $seq2 =~tr/atcgATCG/tagcTAGC/;
                                                                                $seq2 = reverse $seq2;
                                                                        }
                                                                        print data3 "$seq1\t$seq2\t$fakej[0]\t$fakej[1]\t";
                                                                        #print data3 substr($gene{$chr},$fakej-$ban,$bp)."\t";
                                                                        
                                                                        $long = abs($fakej[0]-$fakej[1]);
                                                                        print data3 "0\t$long\n";
                                                                        $fake_i = $fake_i+1;
                                                                        if($fake_i > $num/2)
                                                                        {
                                                                                last;
                                                                        }
                                                                }
                                                                #清空fake用到的变量
                                                                @pre_exon = ();
                                                                @beh_exon = ();
                                                                %fake_exon_unit = ();


                                                                #data1 的 3 标签 循环两边前一个在后面的差 我用那个 @array units 了
                                                                #死循环？？？？？？
                                                                #print "我到死循环\t1\t了\n";

                                                                @exon_data1 = keys %exon_site;
                                                                $count=0;
                                                                @exon3 = ();
                                                                while($count < @exon_data1)
                                                                {
                                                                        @aa = split("_",@exon_data1[$count]);
                                                                        push (@exon3,@aa);
                                                                        $count++;
                                                                }
                                                                $count=0;
                                                                $step =0;
                                                                $numb = @exon3;
                                                                #print "我到死循环\t2\t了\n'$numb=' $numb";
                                                                while($count < $numb)
                                                                {
                                                                        $site=int(rand($j-$i-1))+1+$i;
                                                                        $step++;
                                                                        if($step>50)
                                                                        {
                                                                                #print "我运行了50次不想动了\n";
                                                                                last;
                                                                        }
                                                                        for ($split3 =0;$split3<$numb;$split3+=2)
                                                                        {
                                                                                if($mode == 0)
                                                                                {       if($site > $exon3[$split3] && $site < $exon3[$split3+1]) 
                                                                                        {
                                                                                                #print "我找到了一个 非 3号位点\n";
                                                                                                last;
                                                                                        }
                                                                                }
                                                                                else
                                                                                {
                                                                                        if($site < $exon3[$split3] && $site > $exon3[$split3+1]) 
                                                                                        {
                                                                                                #print "我找到了一个 非 3号位点\n";
                                                                                                last;
                                                                                        }
                                                                                }
                                                                        }
                                                                        if($split3 == $numb)
                                                                        {
                                                                                $seq = substr($gene{$chr},$site-$ban,$bp);
                                                                                
                                                                                if($mode == -1)
                                                                                {
                                                                                        $seq =~tr/atcgATCG/tagcTAGC/;
                                                                                        $seq = reverse $seq;
                                                                                }
                                                                                #print data1 substr($gene{$chr},$site-$ban,$bp)."\t3\n";
                                                                                print data1 "$seq\t$site\t3\n";
                                                                                $count++;
                                                                        }
                                                                }
                                                                
                                                                $line = $exon;
                                                                goto G;
                                                        }
                                                        if ($exon_words[2] eq 'transcript')
                                                        {
                                                                $tran = $exon;
                                                                goto T;
                                                        }
                                                }

                                        }

                                }

                        }
                        else 
                        {
                                last;
                        }
                }
        }
}

